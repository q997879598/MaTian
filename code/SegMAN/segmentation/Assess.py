import os
import cv2
import torch
import numpy as np
import math
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont  # 引入PIL处理中文
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv import Config

# ================= 配置区 =================
CHECKPOINT_FILE = '/root/autodl-tmp/SegMAN/segmentation/outputs/segman_maritime_v2/best_mIoU_iter_30000.pth'
CONFIG_FILE = '/root/autodl-tmp/SegMAN/segmentation/local_configs/segman/base/segman_maritime_b_v2.py'
INPUT_DIR = 'data/my_images'
OUTPUT_DIR = 'outputs/My_results_Chinese'
FONT_PATH = 'font.ttf'  # ⚠️请务必上传一个中文字体文件到同级目录，否则中文会显示方框

# 物理参数
CAMERA_HEIGHT = 5.0      # 相机高度
V_FOV = 45.0
H_FOV = 60.0

# 风险阈值 (单位: 米)
DIST_CRITICAL = 100.0  # 紧急
DIST_WARNING = 300.0   # 警告

# 语义类别
CLASS_OBSTACLE = 0
CLASS_WATER = 1
CLASS_SKY = 2

# 配色 (BGR)
COLOR_CRITICAL = (0, 0, 255)    # 红
COLOR_WARNING = (0, 165, 255)   # 橙
COLOR_SAFE = (0, 255, 0)        # 绿
# ============================================

def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=20):
    """
    工具函数：使用PIL在OpenCV图片上绘制中文
    """
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img)
    # 加载字体，如果找不到则使用默认（可能不支持中文）
    try:
        font = ImageFont.truetype(FONT_PATH, text_size)
    except:
        print(f"⚠️ 警告: 未找到字体文件 {FONT_PATH}，中文可能无法显示。")
        font = ImageFont.load_default()
        
    draw.text(position, text, fill=text_color[::-1], font=font) # RGB转BGR
    
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

class MaritimeSafetySystem:
    def __init__(self, cam_h=CAMERA_HEIGHT, v_fov=V_FOV):
        self.cam_h = cam_h
        self.v_fov = v_fov
        self.radar_size = 600

    def _get_adaptive_horizon(self, mask):
        """修正版地平线检测：寻找最远水面"""
        h, w = mask.shape
        water_mask = (mask == CLASS_WATER).astype(np.uint8)
        
        if np.count_nonzero(water_mask) < 1000:
            return int(h * 0.45), False 

        col_top_indices = np.argmax(water_mask, axis=0)
        valid_cols = (col_top_indices > 0) | (water_mask[0, :] == 1)
        valid_top_ys = col_top_indices[valid_cols]

        if len(valid_top_ys) == 0:
            return int(h * 0.45), False

        horizon_y = int(np.percentile(valid_top_ys, 5))
        horizon_y = max(10, min(horizon_y, h - 10))
        return horizon_y, True

    def _monocular_distance(self, v_pixel, horizon_y, img_h):
        if v_pixel <= horizon_y + 2: return 9999.0
        f_y = (img_h / 2.0) / math.tan(math.radians(self.v_fov / 2.0))
        angle_rad = math.atan((v_pixel - horizon_y) / f_y)
        if angle_rad <= 0: return 9999.0
        dist = self.cam_h / math.tan(angle_rad)
        return dist

    def _quantify_threat(self, dist, bearing_deg):
        """
        威胁度量化评分模型
        Score = (1000 / (dist + 10)) * Weight_Dist + (1 - |bearing|/90) * Weight_Angle
        分数越高，威胁越大
        """
        # 归一化处理
        w_dist = 0.7  # 距离权重更重
        w_angle = 0.3 # 角度权重
        
        # 距离得分 (0~100)
        score_dist = min(100, 5000 / (dist + 1))
        
        # 角度得分 (0~100, 0度最高)
        score_angle = max(0, 100 * (1 - abs(bearing_deg) / 45.0))
        
        total_score = w_dist * score_dist + w_angle * score_angle
        
        if dist < DIST_CRITICAL:
            level = "🔴 严重威胁"
            desc = "紧迫局面"
            color = COLOR_CRITICAL
        elif dist < DIST_WARNING:
            level = "🟠 中度威胁"
            desc = "需关注"
            color = COLOR_WARNING
        else:
            level = "🟢 低威胁"
            desc = "安全监测"
            color = COLOR_SAFE
            
        return total_score, level, desc, color

    def analyze_frame(self, img_shape, pred_mask):
        h, w = img_shape[:2]
        horizon_y, is_adaptive = self._get_adaptive_horizon(pred_mask)
        
        obstacle_mask = (pred_mask == CLASS_OBSTACLE).astype(np.uint8)
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles_data = []
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 50: continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            bottom_y = y + ch
            cx = x + cw // 2
            
            dist = self._monocular_distance(bottom_y, horizon_y, h)
            
            # 计算方位角
            bearing = (cx - w/2) / (w/2) 
            bearing_deg = bearing * (H_FOV / 2)
            
            # 量化威胁
            score, level, desc, color = self._quantify_threat(dist, bearing_deg)
            
            obstacles_data.append({
                'id': i+1,
                'rect': (x, y, cw, ch),
                'dist': dist,
                'bearing': bearing_deg,
                'score': score,
                'level': level,
                'desc': desc,
                'color': color
            })

        # 按威胁分数降序排列 (最危险的排前面)
        obstacles_data.sort(key=lambda x: x['score'], reverse=True)

        # 生成海事建议
        nav_advice = self._generate_maritime_advice(pred_mask, obstacles_data)
        
        return {
            'horizon_y': horizon_y,
            'obstacles': obstacles_data,
            'nav_advice': nav_advice
        }

    def _generate_maritime_advice(self, mask, obstacles):
        """
        基于 COLREGs (国际海上避碰规则) 的建议生成逻辑
        特别是雾航规则：避免左转，除非必要。
        """
        advice = {
            "status": "安全航行",
            "action": "保持航向与航速",
            "reason": "前方航道清爽，无近距离碰撞风险。",
            "detail": []
        }
        
        if not obstacles:
            return advice

        most_dangerous = obstacles[0]
        
        # 1. 判断高危情况
        if most_dangerous['dist'] < DIST_CRITICAL:
            advice['status'] = "🔴 碰撞危险 (COLLISION RISK)"
            
            # 细分场景
            if abs(most_dangerous['bearing']) < 5.0: # 正前方
                advice['action'] = "⚠️ 紧急停船 或 全速倒车"
                advice['reason'] = f"障碍物(ID:{most_dangerous['id']})位于正前方极近距离({most_dangerous['dist']:.1f}m)，转向空间不足。"
            elif most_dangerous['bearing'] < 0: # 障碍物在左舷
                # 根据规则，若对方在左，我方通常保向，但如果是静止障碍物，应向右避让
                advice['action'] = "↪️ 右满舵 (HARD STARBOARD)"
                advice['reason'] = f"障碍物位于左舷船首({abs(most_dangerous['bearing']):.1f}°)，依避碰规则建议向右避让。"
            else: # 障碍物在右舷
                advice['action'] = "🛑 减速并左舵 (如有足够空间)" 
                advice['reason'] = "障碍物位于右舷，威胁极大，建议减速或向左避让(需核实左侧水域)。"
                
        # 2. 判断警告情况
        elif most_dangerous['dist'] < DIST_WARNING:
            advice['status'] = "🟠 紧迫局面 (CLOSE QUARTERS)"
            
            h, w = mask.shape
            # 检查左右水域宽度
            roi = mask[int(h/2):, :]
            left_water = np.sum(roi[:, :int(w/2)] == CLASS_WATER)
            right_water = np.sum(roi[:, int(w/2):] == CLASS_WATER)
            
            if right_water > left_water:
                advice['action'] = "↘️ 建议右转 (Alter Course to Stbd)"
                advice['reason'] = "前方存在潜在威胁，且右侧可航水域较宽，建议右舵避让。"
            else:
                advice['action'] = "↙️ 建议左转 (仅当右侧受阻时)"
                advice['reason'] = "前方存在威胁，右侧水域受限，谨慎左舵避让。"
                
        return advice

    # ================= 渲染部分 (支持中文) =================

    def render_segmentation_view(self, img_raw, pred_mask, analysis_data):
        """生成中文标注的增强视图"""
        vis = img_raw.copy()
        
        # 半透明Mask
        color_mask = np.zeros_like(img_raw)
        color_mask[pred_mask == CLASS_OBSTACLE] = (0, 0, 255) # 红
        color_mask[pred_mask == CLASS_WATER] = (255, 0, 0)    # 蓝
        vis = cv2.addWeighted(vis, 0.7, color_mask, 0.3, 0)
        
        # 画地平线
        h_y = analysis_data['horizon_y']
        cv2.line(vis, (0, h_y), (vis.shape[1], h_y), (255, 200, 0), 2)
        vis = cv2_add_chinese_text(vis, "自适应视平线", (10, h_y-25), (255, 200, 0), 20)
        
        # 画障碍物
        for obs in analysis_data['obstacles']:
            x, y, w, h = obs['rect']
            color = obs['color']
            
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            
            # 文本信息
            info_text = f"ID:{obs['id']} D:{obs['dist']:.0f}m"
            vis = cv2_add_chinese_text(vis, info_text, (x, y-25), color, 18)
            vis = cv2_add_chinese_text(vis, obs['level'].split(' ')[1], (x, y+h+5), color, 18)
            
        # 顶部画建议栏
        advice = analysis_data['nav_advice']
        # 背景条
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 80), (0, 0, 0), -1)
        vis = cv2_add_chinese_text(vis, f"态势: {advice['status']}", (20, 10), (0, 255, 255), 24)
        vis = cv2_add_chinese_text(vis, f"指令: {advice['action']}", (20, 45), (0, 255, 0), 28)
            
        return vis

    def render_radar_view(self, analysis_data):
        """生成雷达图 (保持英文或简单中文即可)"""
        size = self.radar_size
        radar = np.zeros((size, size, 3), dtype=np.uint8)
        cx, cy = size // 2, size - 40
        max_range = DIST_WARNING * 1.5
        scale = (size - 60) / max_range
        
        # 刻度
        for r_m in [DIST_CRITICAL, DIST_WARNING]:
            r_px = int(r_m * scale)
            cv2.circle(radar, (cx, cy), r_px, (50, 50, 50), 1)
            cv2.putText(radar, f"{int(r_m)}m", (cx+5, cy-r_px), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            
        # 扇形
        fov_rad = math.radians(H_FOV/2)
        pt_left = (cx - int(math.sin(fov_rad)*size), cy - int(math.cos(fov_rad)*size))
        pt_right = (cx + int(math.sin(fov_rad)*size), cy - int(math.cos(fov_rad)*size))
        cv2.line(radar, (cx, cy), pt_left, (30, 30, 30), 1)
        cv2.line(radar, (cx, cy), pt_right, (30, 30, 30), 1)
        
        # 本船
        cv2.arrowedLine(radar, (cx, cy+10), (cx, cy-20), (255, 255, 255), 3)
        
        # 目标
        for obs in analysis_data['obstacles']:
            d = min(obs['dist'], max_range)
            angle_rad = math.radians(obs['bearing'])
            rx = cx + int(d * math.sin(angle_rad) * scale)
            ry = cy - int(d * math.cos(angle_rad) * scale)
            cv2.circle(radar, (rx, ry), 8, obs['color'], -1)
            cv2.putText(radar, str(obs['id']), (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            
        return radar

    def generate_text_report(self, filename, analysis_data):
        """生成详细中文分析报告"""
        lines = []
        lines.append(f"====== 船舶视觉环境感知与威胁评估报告 ======")
        lines.append(f"数据源: {filename}")
        lines.append(f"视平线位置: 像素行 {analysis_data['horizon_y']}")
        lines.append(f"-" * 60)
        
        advice = analysis_data['nav_advice']
        lines.append(f"【综合态势】: {advice['status']}")
        lines.append(f"【推荐操作】: {advice['action']}")
        lines.append(f"【决策依据】: {advice['reason']}")
        lines.append(f"-" * 60)
        
        lines.append(f"【目标详细分析表】 (按威胁度排序)")
        lines.append(f"{'ID':<4}{'距离(m)':<10}{'方位':<12}{'威胁等级':<10}{'威胁分':<8}{'状态描述'}")
        
        if not analysis_data['obstacles']:
            lines.append("当前视野范围内未检测到明显水面障碍物。")
        else:
            for obs in analysis_data['obstacles']:
                # 格式化方位
                bear_str = f"右舷 {obs['bearing']:.1f}°" if obs['bearing'] > 0 else f"左舷 {abs(obs['bearing']):.1f}°"
                
                # 简单分析文案
                detail_analysis = ""
                if obs['dist'] < DIST_CRITICAL:
                    detail_analysis = "距离极近，需立即避让"
                elif abs(obs['bearing']) < 2.0:
                    detail_analysis = "正前方存在碰撞风险"
                else:
                    detail_analysis = "保持监控"
                    
                lines.append(f"{obs['id']:<4}{obs['dist']:<10.1f}{bear_str:<12}{obs['level'].split(' ')[1]:<10}{obs['score']:<8.1f}{detail_analysis}")
        
        lines.append("=" * 60)
        lines.append("注: 距离基于单目视觉平面假设估算，仅供辅助参考。")
        return "\n".join(lines)

def main():
    # 0. 检查字体
    if not os.path.exists(FONT_PATH):
        print(f"❌ 错误: 未在当前目录找到字体文件 '{FONT_PATH}'。")
        print("请上传一个中文字体文件 (如 simhei.ttf) 并重命名为 font.ttf，否则中文将无法显示。")
        # return # 注释掉这行可以强行运行，但中文会乱码

    # 1. 文件夹
    dirs = {
        'seg': os.path.join(OUTPUT_DIR, '1_可视化监测图'),
        'radar': os.path.join(OUTPUT_DIR, '2_雷达态势图'),
        'report': os.path.join(OUTPUT_DIR, '3_评估报告')
    }
    for d in dirs.values():
        if not os.path.exists(d): os.makedirs(d)

    # 2. 模型
    cfg = Config.fromfile(CONFIG_FILE)
    if isinstance(cfg.data.test, list): cfg.data.test = cfg.data.test[0]
    model = init_segmentor(cfg, CHECKPOINT_FILE, device='cuda:0')
    
    safety_sys = MaritimeSafetySystem()

    files = glob(os.path.join(INPUT_DIR, '*'))
    files = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
    print(f"📂 开始处理 {len(files)} 张图片...")

    for path in tqdm(files):
        name = os.path.basename(path)
        name_no_ext = os.path.splitext(name)[0]
        img_raw = cv2.imread(path)
        if img_raw is None: continue

        try:
            # 推理
            result = inference_segmentor(model, img_raw)
            pred_mask = result[0]

            # 分析
            analysis_data = safety_sys.analyze_frame(img_raw.shape, pred_mask)

            # 输出1: 中文可视化图
            vis_seg = safety_sys.render_segmentation_view(img_raw, pred_mask, analysis_data)
            cv2.imwrite(os.path.join(dirs['seg'], name), vis_seg)

            # 输出2: 雷达图
            vis_radar = safety_sys.render_radar_view(analysis_data)
            cv2.imwrite(os.path.join(dirs['radar'], f"{name_no_ext}_radar.png"), vis_radar)

            # 输出3: 中文报告
            report_text = safety_sys.generate_text_report(name, analysis_data)
            with open(os.path.join(dirs['report'], f"{name_no_ext}_报告.txt"), 'w', encoding='utf-8') as f:
                f.write(report_text)

        except Exception as e:
            print(f"处理 {name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"✅ 处理完成！结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
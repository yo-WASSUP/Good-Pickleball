import argparse
import os
from importlib.util import find_spec

from pickleball_analysis.system import PickleballAnalysisSystem, load_runtime_dependencies


def validate_cli_dependencies(args, parser):
    if args.player_detector == 'yolo-person' and args.person_tracker != 'none' and find_spec('lap') is None:
        parser.exit(
            2,
            "error: --person-tracker requires lap>=0.5.12.\n"
            "Install it with one of these commands:\n"
            "  pip install -r requirements.txt\n"
            "  pip install \"lap>=0.5.12\"\n"
            "Or disable MOT tracking with: --person-tracker none\n",
        )


def main():
    parser = argparse.ArgumentParser(description='Good-Pickleball match video analysis system')
    parser.add_argument('--video-path', default='videos/demo.mp4', type=str, help='Input video file path')
    parser.add_argument('--template-path', default='templates/demo.png', type=str, help='Court template image path; omit to open a file picker')
    parser.add_argument('--output-dir', default=None, type=str, help='Output directory, defaults to outputs/<video name>')
    parser.add_argument('--ball-model', default='weights/tennis-ball.pt', type=str, help='YOLO ball detector path; defaults to the copied Good-Tennis baseline model')
    parser.add_argument('--player-detector', default='pose', choices=['pose', 'yolo-person'], help='Player detector: pose keypoints or YOLO person boxes')
    parser.add_argument('--person-model', default='weights/yolo26s.pt', type=str, help='YOLO person detector path or model name')
    parser.add_argument('--person-tracker', default='botsort', choices=['none', 'botsort', 'bytetrack'], help='YOLO person multi-object tracker')
    parser.add_argument('--player-detect-interval', default=1, type=int, help='Player detection interval in processed court frames')
    parser.add_argument('--pose-family', default='rtmpose', choices=['rtmpose', 'rtmo', 'yolo-pose'], help='Pose model family')
    parser.add_argument('--pose-mode', default='balanced', choices=['lightweight', 'balanced', 'performance'], help='RTMPose / RTMO model size')
    parser.add_argument('--yolo-pose-model', default='weights/yolo11s-pose.pt', type=str, help='YOLO pose model path or model name')
    parser.add_argument('--court-detection', default='auto-fallback', choices=['manual', 'auto', 'auto-fallback'], help='Court corner detection mode')
    parser.add_argument('--pose-roi', choices=['true', 'false'], default='true', help='Show the pose detection ROI')
    parser.add_argument('--display', choices=['true', 'false'], default='true', help='Show the OpenCV preview window')
    parser.add_argument('--skeletons', choices=['true', 'false'], default='true', help='Draw human skeletons')
    parser.add_argument('--player-trajectories', choices=['true', 'false'], default='true', help='Draw player trajectories')
    parser.add_argument('--court-trajectory', choices=['true', 'false'], default='true', help='Draw court trajectory overlay')
    parser.add_argument('--pickleball-trajectory', '--tennis-ball-trajectory', dest='ball_trajectory', choices=['true', 'false'], default='true', help='Draw pickleball trajectory')
    parser.add_argument('--bounce-detection', choices=['true', 'false'], default='true', help='Detect and annotate bounce candidates')
    parser.add_argument('--bounce-classifier', default='', type=str, help='Optional bounce classifier pkl path; empty uses rule scoring')
    parser.add_argument('--mini-map', choices=['true', 'false'], default='true', help='Draw the court mini map')
    parser.add_argument('--player-stats', choices=['true', 'false'], default='true', help='Draw player movement stats')
    parser.add_argument('--save-images', action='store_true', default=False, help='Save processed frames')
    parser.add_argument('--performance-stats', action='store_true', default=False, help='Print performance timings')
    parser.add_argument('--visualize-positions', choices=['true', 'false'], default='true', help='Generate player heatmaps and scatter plots')
    parser.add_argument('--audio', choices=['true', 'false'], default='true', help='Keep original audio in the output video')
    parser.add_argument('--language', default='zh', choices=['zh', 'en'], help='Overlay language')
    args = parser.parse_args()
    validate_cli_dependencies(args, parser)

    load_runtime_dependencies()

    if args.language == 'en':
        from pickleball_analysis.visualization.player_positions_en import analyze_player_positions
    else:
        from pickleball_analysis.visualization.player_positions_zh import analyze_player_positions

    system = PickleballAnalysisSystem(
        args.video_path,
        show_display=args.display == 'true',
        show_skeletons=args.skeletons == 'true',
        show_player_trajectories=args.player_trajectories == 'true',
        show_court_trajectory=args.court_trajectory == 'true',
        show_ball_trajectory=args.ball_trajectory == 'true',
        show_player_stats=args.player_stats == 'true',
        show_performance_stats=args.performance_stats,
        save_images=args.save_images,
        language=args.language,
        output_dir=args.output_dir,
        ball_model_path=args.ball_model,
        template_path=args.template_path,
        pose_mode=args.pose_mode,
        pose_family=args.pose_family,
        yolo_pose_model=args.yolo_pose_model,
        player_detector=args.player_detector,
        person_model=args.person_model,
        person_tracker=args.person_tracker,
        player_detect_interval=args.player_detect_interval,
        show_pose_roi=args.pose_roi == 'true',
        court_detection=args.court_detection,
        show_bounce_detection=args.bounce_detection == 'true',
        bounce_classifier_path=args.bounce_classifier,
        show_mini_map=args.mini_map == 'true',
    )

    system.keep_audio = args.audio == 'true'
    system.process_video()

    if args.visualize_positions == 'true':
        print("\nGenerating player position visualizations...")
        analyze_player_positions(system.detections_path, os.path.join(system.save_dir, 'position_visualizations'), fps=system.fps)
        print("Player position visualizations completed")


if __name__ == "__main__":
    main()
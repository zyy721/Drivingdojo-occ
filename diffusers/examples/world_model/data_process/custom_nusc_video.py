import os
import argparse
import shutil
from nuscenes.nuscenes import NuScenes
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description='Export NuScenes video frames.')
    parser.add_argument('--version', type=str, choices=['trainval', 'test'], default='trainval', 
                        help='Specify the NuScenes version to use (train or test).')
    parser.add_argument('--nusc_root', type=str, default='/mnt/vdb1/nuscenes')
    parser.add_argument('--sensors', type=str, nargs='+', default=['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 
                                                                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'], 
                        help='List of camera sensors to export (e.g., CAM_FRONT CAM_BACK).')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save exported video frames.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Set NuScenes root directory
    nusc = NuScenes(version=f'v1.0-{args.version}', dataroot=args.nusc_root, verbose=True)

    # Load scene tokens
    scene_tokens = [s['token'] for s in nusc.scene]

    # Create output directory
    # os.makedirs(args.output_dir, exist_ok=True)

    all_cam_video_dict = {}

    # Iterate over selected sensors
    for sensor in args.sensors:
        video_dict = {}

        for scene_token in scene_tokens:
            scene = nusc.get('scene', scene_token)
            name = scene['name']
            # os.makedirs(os.path.join(args.output_dir, name), exist_ok=True)
            scene_imgs = []

            first_sample_token = scene['first_sample_token']
            last_sample_token = scene['last_sample_token']

            first_sample_rec = nusc.get('sample', first_sample_token)
            last_sample_rec = nusc.get('sample', last_sample_token)

            sample_token = first_sample_token
            sample_token = scene['first_sample_token']
            sample = nusc.get('sample', sample_token)
            token = sample['data'][sensor]
            # os.makedirs(os.path.join(args.output_dir,name,sensor), exist_ok=True)

            while token != '':
                data = nusc.get('sample_data', token)
                path_scene_img = os.path.join(args.nusc_root, data['filename'])
                scene_imgs.append(path_scene_img)
                token = data["next"]
            
            video_dict[name] = scene_imgs

            # for img in scene_imgs:
            #     shutil.copy(os.path.join(args.nusc_root,img), os.path.join(args.output_dir,name,sensor,img.split('/')[-1]))
            print('finish', name)

        all_cam_video_dict[sensor] = video_dict

    with open('../data/nusc_video_all_cam.pkl', 'wb') as f:
        pickle.dump(all_cam_video_dict, f)


if __name__ == '__main__':
    main()

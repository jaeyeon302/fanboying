import facefinder
import videoEditor
import os, time
import argparse

__TOLERANCE = 0.4
__STD_PROCESS_VIDEO_WIDTH = 320
__ANALYZING_FRME_DELTA = 1

def main(input_video_file_name, target_face_img_file_name, output_video_file_name):
    start_time = time.time()

    video_meta = videoEditor.Video_meta(input_video_file_name)
    matched_frame_sections = facefinder.find_target_face_multiprocess(
        video_meta,
        target_face_img_file_name,
        __STD_PROCESS_VIDEO_WIDTH,
        __ANALYZING_FRME_DELTA,
        __TOLERANCE
    )
    matched_time_sections = videoEditor.convert_section_from_frames_to_secs(
        matched_frame_sections,
        video_meta.fps
    )

    tmp_video_file_name = "tmp.mp4"
    tmp_audio_file_name = "tmp.mp3"

    videoEditor.cv2_save_frames_multiprocess(
        video_meta,
        matched_frame_sections,
        tmp_video_file_name
    )

    videoEditor.trim_original_audio_to_audio_subclips(
        video_meta.file_name,
        matched_time_sections,
        tmp_audio_file_name
    )

    videoEditor.ffmpeg_merge_video_audio(
        tmp_video_file_name,
        tmp_audio_file_name,
        output_video_file_name
    )

    os.remove(tmp_audio_file_name)
    os.remove(tmp_video_file_name)
    end_time = time.time()

    output_Video_meta = videoEditor.Video_meta(output_video_file_name)
    print("original video length : {} sec".format(video_meta.frame_length*video_meta.fps))
    print("result video length : {} sec".format(output_Video_meta.frame_length*output_Video_meta.fps))
    print("processing time : {} sec".format(end_time-start_time))
    print("done!")


def test():
    INPUT_VIDEO_FILE_NAME = "jangiha.mp4"
    OUTPUT_VIDEO_FILE_NAME = "jangiha_module_test.mp4"
    TARGET_FACE_IMG_FILE_NAME = "jangiha.jpg"

    main(INPUT_VIDEO_FILE_NAME,TARGET_FACE_IMG_FILE_NAME,OUTPUT_VIDEO_FILE_NAME)
    
def expand_path(path):
    return os.path.abspah(os.path.expanduser(path))

def existed_file_path(path):
    #reference : https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
    path = expand_path(path)
    if os.path.isfile(path):
        return path
    else:
        raise IOError(path)

def not_existed_file_path(path):
    path = expand_path(path)
    if not os.path.isfile(path):
        return path
    else:
        raise IOError(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="fanboying", description='edit the video for my bias')
    parser.add_argument("--input-video", "-i", type=existed_file_path,help="video file path to process")
    parser.add_argument("--my-bias-img", "-b", type=existed_file_path,help="img file having your bias's face")
    parser.add_argument("--output-video","-o", type=not_existed_file_path,help="output video file name")
    
    args = parser.parse_args()
    main(args.input_video, args.my_bias_img, args.output_video)




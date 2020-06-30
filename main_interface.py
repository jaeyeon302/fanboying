import facefinder
import videoEditor
import os

INPUT_VIDEO_FILE_NAME = "jangiha.mp4"
OUTPUT_VIDEO_FILE_NAME = "jangiha_module_test.mp4"
TARGET_FACE_IMG_FILE_NAME = "jangiha.jpg"

__TOLERANCE = 0.4
__STD_PROCESS_VIDEO_WIDTH = 320
__ANALYZING_FRME_DELTA = 1

if __name__ == "__main__":
    video_meta = videoEditor.Video_meta(INPUT_VIDEO_FILE_NAME)
    matched_frame_sections = facefinder.find_target_face_multiprocess(
        video_meta,
        TARGET_FACE_IMG_FILE_NAME,
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
        OUTPUT_VIDEO_FILE_NAME
    )

    os.remove(tmp_audio_file_name)
    os.remove(tmp_video_file_name)


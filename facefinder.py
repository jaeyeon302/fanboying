import face_recognition
import cv2
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips

TARGET_FACE_IMG_PATH = "./jihyo.jpg"#"./jeon.jpg"
OUTPUT_VIDEO_PATH = "fancy_jihyo.mp4"#"./test.mp4"
INPUT_VIDEO_PATH = "fancy_test.mp4"#"./jeon.mp4"

__TOLERANCE = 0.4
__STD_PROCESS_VIDEO_WIDTH = 320
__SUB_CLIPS_DIR = "./subclips"


def find_target_face(input_video,target_face_encoding, 
                    std_process_video_width=320, 
                    analyzing_frame_delta=1,
                    tolerance=0.4):

    is_saved = False
    
    video_frame_length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))//4
    video_fps = input_video.get(cv2.CAP_PROP_FPS)
    video_frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))

    resizing_scale = std_process_video_width/video_frame_width
    if resizing_scale < 1:
        resize = lambda frame: cv2.resize(frame,dsize=(0,0),fx=resizing_scale,fy=resizing_scale,interpolation=cv2.INTER_AREA)
    else:
        resize = lambda frame: frame

    def check_frame_has_target_face(frame):
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
        match = face_recognition.compare_faces(face_encodings, target_face_encoding, tolerance=tolerance)
        if sum(match):return True
        else: return False

    interest_sections = []
    section_start = 0
    if analyzing_frame_delta==1:
        for frame_number in range(0,video_frame_length):
            print("{} / {}".format(frame_number,video_frame_length))
            ret, frame = input_video.read()
            resized_frame = resize(frame)
            rgb_frame = resized_frame[:, :, ::-1]
            matched = check_frame_has_target_face(rgb_frame)

            if matched and not is_saved:
                print("My bias appears! at {} frame".format(frame_number))
                is_saved = True
                section_start = frame_number
            elif not matched and is_saved:
                print("My bias disappears! at {} frame".format(frame_number))
                is_saved = False
                interest_sections.append((section_start,frame_number))
    else:
        for frame_number in range(0, video_frame_length, analyzing_frame_delta):
            print("{} / {}".format(frame_number,video_frame_length))
            input_video.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
            ret, frame = input_video.read()
            resized_frame = resize(frame)
            rgb_frame = resized_frame[:, :, ::-1]
            matched = check_frame_has_target_face(rgb_frame)

            if matched and not is_saved:
                print("My bias appears! at {} frame".format(frame_number))
                is_saved = True
                section_start = frame_number
            elif not matched and is_saved:
                print("My bias disappears! at {} frame".format(frame_number))
                is_saved = False
                interest_sections.append((section_start,frame_number))

    return interest_sections

def convert_section_from_frames_to_secs(section_list,fps,round_num=None):
    if round_num == None:
        return [(s/fps,e/fps) for (s,e) in section_list]
    else:
        return [(round(s/fps,round_num), round(e/fps,round_num)) for (s,e) in section_list]


def cv2_save_frames(input_video,interest_sections,output_file_name):
    fps = input_video.get(cv2.CAP_PROP_FPS)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height))
    idx = 0
    for (s,e) in interest_sections:
        input_video.set(cv2.CAP_PROP_POS_FRAMES,s)
        print("{} / {} clips saved".format(idx+1,len(interest_sections)))
        for frame_number in range(s,e):
            ret,frame = input_video.read()
            output_video.write(frame)
        idx+=1

def ffmpeg_merge_video_audio(video_file_name,audio_file_name,output_file_name):
    #ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac output.mp4

    cmd = [get_setting("FFMPEG_BINARY"),"-y",
            "-i",video_file_name,
            "-i",audio_file_name,
            "-c:v", "copy",
            "-c:a", "aac",
            output_file_name]
    print(cmd)
    subprocess_call(cmd)


if __name__ == "__main__":
    
    #load video
    input_video = cv2.VideoCapture(INPUT_VIDEO_PATH)

    #load target face(my bias)
    target_face_img = face_recognition.load_image_file(TARGET_FACE_IMG_PATH)
    target_face_encoding = face_recognition.face_encodings(target_face_img)[0]

    matched_frame_section = find_target_face(input_video,target_face_encoding,
                                        std_process_video_width=__STD_PROCESS_VIDEO_WIDTH,
                                        analyzing_frame_delta=1,
                                        tolerance=__TOLERANCE)
    matched_time_section = convert_section_from_frames_to_secs(matched_frame_section, input_video.get(cv2.CAP_PROP_FPS))

    tmp_video_file = "tmp.mp4"
    tmp_audio_file = "tmp.mp3"
    cv2_save_frames(input_video,matched_frame_section,tmp_video_file)
   
    original_audio = AudioFileClip(INPUT_VIDEO_PATH)
    audio_subclips = [original_audio.subclip(s,e) for (s,e) in matched_time_section]
    edited_audio = concatenate_audioclips(audio_subclips)
    edited_audio.write_audiofile(tmp_audio_file)
    ffmpeg_merge_video_audio(tmp_video_file ,tmp_audio_file, OUTPUT_VIDEO_PATH)

    os.remove(tmp_video_file)
    os.remove(tmp_audio_file)

    input_video.release()

    


import face_recognition
import cv2
import os
from collections import namedtuple as nt
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips
import multiprocessing
import time

TARGET_FACE_IMG_PATH = "./jeon.jpg"         #picture of my bias
OUTPUT_VIDEO_PATH = "./only_jeon_jtbc_multi2.mp4" #name of edited video
INPUT_VIDEO_PATH = "./jeon_test2.mp4"             #name of 

__ANALYZING_FRAME_DELTA = 1
__TOLERANCE = 0.4
__STD_PROCESS_VIDEO_WIDTH = 320
#fixed frame width to process. Frames' height will be resized proportionally


def find_target_face(input_video_file_name,frame_range):
    """
    input_video_file_name : (str) original input video file name
    frame_range : (tuple) (start, end) - frame range to process in this funciton.
    """
    global analyzing_frame_delta, target_face_encoding, tolerance

    input_video = cv2.VideoCapture(input_video_file_name)
    #video_frame_length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #resize frame proportionally if frame's width > std process video width
    if resizing_scale < 1:
        resize = lambda frame: cv2.resize(frame,dsize=(0,0),fx=resizing_scale,fy=resizing_scale,interpolation=cv2.INTER_AREA)
    else:
        resize = lambda frame: frame

    interest_sections = []  #contain video section where the target face appeared
    saving = False          #indicator whether the target face is appearing or not
    section_start = 0       
    
    s,e = frame_range
    input_video.set(cv2.CAP_PROP_POS_FRAMES,s) #set the first frame 

    #let's fanboying
    for frame_number in range(s,e,analyzing_frame_delta):
        print("PID:{}, {}%".format(os.getpid(),round(100*(frame_number-s)/(e-s),2)))

        #if the frames to skip to analyze are more than 0
        if analyzing_frame_delta > 1:
            input_video.set(cv2.CAP_PROP_POS_FRAMES,frame_number)

        ret, frame = input_video.read()
        #BGR color space(opencv used) -> RGB color space(face_recognition used)
        frame = frame[:, :, ::-1]
        resized_frame = resize(frame)
        
        #find face location using resized small frame
        face_locations = face_recognition.face_locations(resized_frame) #the most heavy job

        #encoding is not the heavy task. so Used original frame size
        face_locations = [tuple([int(pos/resizing_scale) for pos in loc]) for loc in face_locations]
        face_encodings = face_recognition.face_encodings(frame,face_locations)  #the lightest job
        match = face_recognition.compare_faces(face_encodings, target_face_encoding, tolerance=tolerance) #similar with encoding

        matched = False if sum(match) == 0 else True

        if matched and not saving:
            print("My bias appears! at {} frame".format(frame_number))
            saving = True
            section_start = frame_number
        elif not matched and saving:
            print("My bias disappears! at {} frame".format(frame_number))
            saving = False
            interest_sections.append((section_start, frame_number))

    return interest_sections

def child_initializer(_target_face_encoding, _resizing_scale, _analyzing_frame_delta, _tolerance):
    """
    Initialize common variables for the multiprocessing jobs
    """
    global target_face_encoding, resizing_scale, analyzing_frame_delta, tolerance

    target_face_encoding = _target_face_encoding
    analyzing_frame_delta = _analyzing_frame_delta
    resizing_scale = _resizing_scale
    tolerance = _tolerance

def main(target_face_img_name, input_video_file_name, output_file_name, std_process_video_width, analyzing_frame_delta, tolerance):
    #load target face(my bias)
    target_face_img = face_recognition.load_image_file(target_face_img_name)
    target_face_encoding = face_recognition.face_encodings(target_face_img)[0]

    #load video and get video frame information
    input_video = cv2.VideoCapture(input_video_file_name)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_length = input_video.get(cv2.CAP_PROP_FRAME_COUNT) 
    frame_width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    input_video.release()

    #divide jobs
    num_of_cpu = multiprocessing.cpu_count()
    frame_per_cpu = frame_length // num_of_cpu
    frame_ranges = [(frame_per_cpu*(cpu_idx),frame_per_cpu*(cpu_idx+1)) for cpu_idx in range(0,num_of_cpu-1) ]
    frame_ranges.append((frame_per_cpu*(num_of_cpu-1), frame_length))
    frame_ranges = [ (int(s),int(e)) for s,e in frame_ranges]

    #define resizing scale to process in fixed frame size
    resizing_scale = std_process_video_width/frame_width

    #start multiprocessing!
    #reference1 : https://pythonspeed.com/articles/python-multiprocessing/ # 왜 내 pool은 멈출까...
    #reference2 : https://stackoverflow.com/questions/25825995/python-multiprocessing-only-one-process-is-running 
    matched_frame_sections = []
    with multiprocessing.get_context("spawn").Pool(num_of_cpu, initializer = child_initializer, 
                                initargs=(target_face_encoding, resizing_scale, analyzing_frame_delta, tolerance)) as pool:
        results = [pool.apply_async(find_target_face,(input_video_file_name, frame_range)) for frame_range in frame_ranges]
        for res in results:
            matched_frame_sections.extend(res.get())
    
    matched_frame_sections = expand_sections(matched_frame_sections, 0)

    matched_time_sections = convert_section_from_frames_to_secs(matched_frame_sections,fps)
    tmp_video_file_name = "tmp.mp4"
    tmp_audio_file_name = "tmp.mp3"

    #generate edited video and audo then merge them
    #cv2_save_frames(input_video_file_name, matched_frame_sections, tmp_video_file_name)
    cv2_save_frames_multi(input_video_file_name, matched_frame_sections, tmp_video_file_name, num_of_cpu)
    trim_audio_to_audio_subclips(input_video_file_name, matched_time_sections, tmp_audio_file_name)
    ffmpeg_merge_video_audio(tmp_video_file_name, tmp_audio_file_name, output_file_name)

    os.remove(tmp_video_file_name)
    os.remove(tmp_audio_file_name)

    
def expand_sections(frame_section_list, frame_number_to_expand = 2):
    if frame_number_to_expand == 0 :
        return frame_section_list
    expanded_sections = [(s-frame_number_to_expand, e + frame_number_to_expand) for s,e in frame_section_list]
    merged_sections = [expanded_sections[0]]
    last_e = 0
    for section, next_section in zip(expanded_sections[:-1], expanded_sections[1:]):
        s,e = section
        next_s, next_e = next_section
        last_s, last_e = merged_sections[-1]
        merged = None
        if e >= next_s:
            merged = (s,next_e)
        
        if merged:
            merged_sections[-1] = (last_s, next_e)
        else:
            merged_sections.append(next_section)

    return merged_sections

def convert_section_from_frames_to_secs(section_list,fps,round_num=None):
    """
    convert sections' unit from frame number to seconds
    """
    if round_num == None:
        return [(s/fps,e/fps) for (s,e) in section_list]
    else:
        return [(round(s/fps,round_num), round(e/fps,round_num)) for (s,e) in section_list]

def cv2_save_frames_multi(input_video_file_name,interest_sections,output_file_name,num_of_cpu):
    section_per_cpu = len(interest_sections)//num_of_cpu
    section_ranges = [ (section_per_cpu*cpu_idx, section_per_cpu*(cpu_idx+1)) for cpu_idx in range(num_of_cpu-1) ]
    section_ranges.append( (section_per_cpu*(num_of_cpu-1), len(interest_sections)) )
    sections = [ interest_sections[s:e] for s,e in section_ranges ]
    output_names = [ "{}_subclips.mp4".format(idx) for idx in range(num_of_cpu) ]

    with multiprocessing.get_context("spawn").Pool(num_of_cpu) as pool:
        results = [pool.apply_async(cv2_save_frames, (input_video_file_name, section, output_name) ) for (section, output_name) in zip(sections,output_names)]
        res = [res.get() for res in results]
        
    ffmpeg_merge_video_without_reencoding(output_names,output_file_name)
    for name in output_names:
        os.remove(name)    
    
def cv2_save_frames(input_video_file_name, interest_sections, output_file_name):
    """
    save frame by frame
    """
    input_video = cv2.VideoCapture(input_video_file_name)

    fps = input_video.get(cv2.CAP_PROP_FPS)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height))
    idx = 0
    for s,e in interest_sections:
        #calibrate the frame range exceeding the original video frame counts
        #This exceeding was occured during expanding the frame sections
        if s < 0 : s = 0
        elif e > frame_length : e = frame_length

        input_video.set(cv2.CAP_PROP_POS_FRAMES,s)
        print("PID:{}, {} / {} clips saved".format(os.getpid(), idx+1,len(interest_sections)))
        for frame_number in range(s,e):
            ret,frame = input_video.read()
            output_video.write(frame)
        idx+=1
    input_video.release()

def trim_audio_to_audio_subclips(input_video_file_name,matched_time_section,output_file_name):
    """
    save edited audio where my bias appeared in video
    """
    audio = AudioFileClip(input_video_file_name)
    audio_subclips = [audio.subclip(s,e) for (s,e) in matched_time_section]
    editted_audio = concatenate_audioclips(audio_subclips)
    editted_audio.write_audiofile(output_file_name)

def ffmpeg_merge_video_without_reencoding(subclip_list,output_file_name):
    #ffmpeg -f concat -safe 0 -i vidlist.txt -c copy output
    contents = ""
    for name in subclip_list:
        contents += "file '{}'\n".format(name)

    tmp_video_list_file = "tmp_subclips.txt"
    with open(tmp_video_list_file,"w") as f:
        f.write(contents)
        
    cmd = [get_setting("FFMPEG_BINARY"),"-y",
            "-f", "concat",
            "-safe", "0",
            "-i", tmp_video_list_file,
            "-c","copy",
            output_file_name
    ]
    subprocess_call(cmd)
    os.remove(tmp_video_list_file)

   
def ffmpeg_merge_video_audio(video_file_name,audio_file_name,output_file_name):
    """
    run this command
    $ ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac output.mp4
    video.mp4 -> video_file_name
    audio.wav -> audio_file_name
    output.mp4 -> output_file_name
    """
    
    cmd = [get_setting("FFMPEG_BINARY"),"-y",
            "-i",video_file_name,
            "-i",audio_file_name,
            "-c:v", "copy",
            "-c:a", "aac",
            output_file_name]

    subprocess_call(cmd) #from moviepy

if __name__ =="__main__":
    start = time.time()
    main(TARGET_FACE_IMG_PATH, INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, __STD_PROCESS_VIDEO_WIDTH, __ANALYZING_FRAME_DELTA, __TOLERANCE)
    print("Processing Time: {}".format(time.time()-start))


    
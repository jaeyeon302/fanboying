import os
import cv2
import multiprocessing
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips

class Video_meta():
    def __init__(self,video_file_name):
        self.file_name = video_file_name
        video = cv2.VideoCapture(video_file_name)
        self.fps = int(video.get(cv2.CAP_PROP_FPS))
        self.frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (w,h)
        video.release()


def expand_frame_sections(frame_section_list, frame_number_to_expand = 2):

    if frame_number_to_expand == 0:
        return frame_section_list

    # add frame margins before and after the frame section
    expanded_sections = [(s-frame_number_to_expand, e + frame_number_to_expand) for s,e in frame_section_list]
    merged_sections = [expanded_sections[0]] # frame sections that duplicated frames are merged
    for section, next_section in zip(expanded_sections[:-1], expanded_sections[1:]):
        s,e = section
        next_s, next_e =next_section
        last_s, last_e = merged_sections[-1]
        merged = None
        if e >= next_s:
            merged = (s,next_e)
        
        if merged:
            #next_section has frmaes duplicated with last section
            merged_sections[-1] = (last_s,next_e)
        else:
            #No duplicated frames
            merged_sections.append(next_section)
    return merged_sections

def convert_section_from_frames_to_secs(section_list, fps, round_num=None):
    """
    convert sections' unit from frame number to seconds
    """
    if round_num == None:
        return [(s/fps,e/fps) for (s,e) in section_list]
    else:
        return [(round(s/fps,round_num), round(e/fps,round_num)) for (s,e) in section_list]

def cv2_save_frames_multiprocess(video_meta, 
                                interest_sections, 
                                output_file_name, 
                                num_of_cpu = multiprocessing.cpu_count()):
    section_per_cpu = len(interest_sections)//num_of_cpu
    section_ranges = [ (section_per_cpu*cpu_idx, section_per_cpu*(cpu_idx+1)) for cpu_idx in range(num_of_cpu-1) ]
    section_ranges.append( (section_per_cpu*(num_of_cpu-1), len(interest_sections)) )
    sections = [ interest_sections[s:e] for s,e in section_ranges ]
    output_names = [ "{}_subclips.mp4".format(idx) for idx in range(num_of_cpu) ]

    with multiprocessing.get_context("spawn").Pool(num_of_cpu) as pool:
        results = [pool.apply_async(cv2_save_frames, (video_meta, section, output_name) ) for (section, output_name) in zip(sections,output_names)]
        res = [res.get() for res in results]
        
    ffmpeg_merge_video_without_reencoding(output_names,output_file_name)
    for name in output_names:
        os.remove(name)   

def cv2_save_frames(video_meta, interest_sections, output_file_name):
    """
    save frame by frame
    """
    input_video = cv2.VideoCapture(video_meta.file_name)
    fps = video_meta.fps
    width, height = video_meta.frame_size
    frame_length = video_meta.frame_length

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

def trim_original_audio_to_audio_subclips(input_video_file_name,matched_time_section,output_file_name):
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
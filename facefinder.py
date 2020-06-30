import face_recognition
import cv2
import multiprocessing
import os

global target_face_encoding, resizing_scale, analyzing_frame_delta, tolerance


def child_initializer(_target_face_encoding, _resizing_scale, _analyzing_frame_delta, _tolerance):
    """
    Initialize common variables for the multiprocessing jobs
    """
    #common variables for find_target_face function
    global target_face_encoding, resizing_scale, analyzing_frame_delta, tolerance

    target_face_encoding = _target_face_encoding
    analyzing_frame_delta = _analyzing_frame_delta
    resizing_scale = _resizing_scale
    tolerance = _tolerance

def find_target_face_multiprocess(video_meta,
                                target_face_img_file_name,
                                std_process_video_width, 
                                analyzing_frame_delta, 
                                tolerance,
                                num_of_cpu=multiprocessing.cpu_count()):
    #load target_face_img
    target_face_img = face_recognition.load_image_file(target_face_img_file_name)
    target_face_encoding = face_recognition.face_encodings(target_face_img)[0]

    
    #divide jobs
    frame_per_cpu = video_meta.frame_length // num_of_cpu
    frame_ranges = [(frame_per_cpu*(cpu_idx),frame_per_cpu*(cpu_idx+1)) for cpu_idx in range(0,num_of_cpu-1) ]
    frame_ranges.append((frame_per_cpu*(num_of_cpu-1), video_meta.frame_length))
    frame_ranges = [ (int(s),int(e)) for s,e in frame_ranges]

    #define resizing scale to process frames in fixed size
    width, height = video_meta.frame_size
    resizing_scale = std_process_video_width/width

    #start multiprocessing!
    #reference1 : https://pythonspeed.com/articles/python-multiprocessing/ # 왜 내 pool은 멈출까...
    #reference2 : https://stackoverflow.com/questions/25825995/python-multiprocessing-only-one-process-is-running 
    matched_frame_sections = []
    with multiprocessing.get_context("spawn").Pool(num_of_cpu, initializer = child_initializer, 
                                initargs=(target_face_encoding, resizing_scale, analyzing_frame_delta, tolerance)) as pool:
        results = [pool.apply_async(find_target_face,(video_meta.file_name, frame_range)) for frame_range in frame_ranges]
        for res in results:
            matched_frame_sections.extend(res.get())
    return matched_frame_sections

def find_target_face(input_video_file_name, frame_range_to_process):
    global target_face_encoding, resizing_scale, analyzing_frame_delta, tolerance

    input_video = cv2.VideoCapture(input_video_file_name)

    if resizing_scale < 1:
        resize = lambda frame: cv2.resize(frame,dsize=(0,0),fx=resizing_scale,fy=resizing_scale,interpolation=cv2.INTER_AREA)
        expand_coordinate_to_original_size = lambda coordinates: [tuple([int(pos/resizing_scale) for pos in loc]) for loc in coordinates]
    else:
        resize = lambda frame: frame
        expand_coordinate_to_original_size = lambda coordinates: coordinates

    interest_video_section = []
    saving = False
    section_start = 0

    s,e = frame_range_to_process
    input_video.set(cv2.CAP_PROP_POS_FRAMES,s)

    #let's fanboying
    for frame_number in range(s,e,analyzing_frame_delta):
        print("PID:{}, {}%".format(os.getpid(),round(100*(frame_number-s)/(e-s),2)))
        if analyzing_frame_delta > 1:
            input_video.set(cv2.CAP_PROP_POS_FRAMES,frame_number)

        ret, frame = input_video.read()
        #BGR color space(opencv used) -> RGB color space(face_recognition used)
        frame = frame[:, :, ::-1]
        resized_frame = resize(frame)
        
        #find face location using resized small frame
        face_locations = face_recognition.face_locations(resized_frame) #the most heavy job

        #encoding is not the heavy task. so Used original frame size
        face_locations = expand_coordinate_to_original_size(face_locations)
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
            interest_video_section.append((section_start, frame_number))

    input_video.release()
    return interest_video_section
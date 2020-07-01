# fanboying
영상에서 최애가 나온 장면만 잘라내어 모아주는 프로그램.   
얼굴이 잘 나온 나의 최애 이미지와 최애가 나온 영상을 넣어주면 최애가 나온 부분만 잘라서 합쳐줍니다!

## 사용 방법
```bash
python3 main.py -i <input_video_filename> -b <bias_img_filename> -o <output_video_filename>
```

## 필요한 패키지
- [face_recognition](https://github.com/ageitgey/face_recognition/blob/master/README_Korean.md) _(ver:1.3.0)_
- [moviepy](https://pypi.org/project/moviepy/) _(ver:1.0.3)_
- [opencv](https://pypi.org/project/opencv-python/) _(ver:4.2.0.34)_

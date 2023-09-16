import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import webbrowser

model=tf.keras.models.load_model(r'D:\Data Science\DL Deep Learning\cricket_shots\shot_model.h5')

image1=Image.open(r'D:\Data Science\DL Deep Learning\cricket_shots\Home_page.jpg')
image2=Image.open(r'D:\Data Science\DL Deep Learning\cricket_shots\coverdrive.jpg')
image3=Image.open(r'D:\Data Science\DL Deep Learning\cricket_shots\flick.jpg')
image4=Image.open(r'D:\Data Science\DL Deep Learning\cricket_shots\sweep.jpg')
image5=Image.open(r'D:\Data Science\DL Deep Learning\cricket_shots\pullshot.jpg')

def main():
    st.sidebar.header("Home")
    selected_section=st.sidebar.radio("",["About","Cricket Shots","Upload"])

    if selected_section=="About":
        home_section()

    elif selected_section=="Cricket Shots":
        shots_section()

    elif selected_section=="Upload":
        upload_section()

    
def home_section():
    st.header("Cricket Shot Predictor")
    st.write("Cricket is a widely popular sport with numerous shots and playing techniques. Analyzing a players performance requires expert judgment to classify shots accurately. This web application provides you with a platform to find the shots played by the players. You needed to upload the images to find the shot played by the batsmen.")
    st.image(image1)


def shots_section():
    st.header("Different Cricket Shots")
    st.write("Different types of Cricket shots are given below")
    st.image(image2,caption='Cover Drive',width=400)
    st.image(image3,caption='Flick Shot',width=400)
    st.image(image4,caption='Sweep Shot',width=400)
    st.image(image5,caption='Pull Shot',width=400)



def predict_new(imgpath,model):

  image_resized=cv2.resize(imgpath,(200,200))
  image_reshaped=image_resized.reshape(1,200,200,3)

  pred=model.predict(image_reshaped)
  x=np.argmax(pred)
  if x==0:
    return ('Cover Drive shot')
  elif x==1:
    return('Flick Shot')
  elif x==2:
    return('Sweep Shot')
  else:
    return('Pullshot')
  
def link_youtube(url):
    try:
        webbrowser.get().open(url)
        return ('Done')
    except:
        return ("check your internet connection")
    

def upload_section():
    st.header("Upload")
    
    uploaded_image=st.file_uploader("upload the image",type=["jpg","png"])
    if uploaded_image is not None:
        file_bytes=np.asarray(bytearray(uploaded_image.read()),dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)
        image=cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
       
        

        st.image(uploaded_image)
     
        result=''
        if st.button('Predicted Cricket Shot'):
            result=predict_new(image,model)
        st.success(result)

        st.write("Related Youtube Searches...")

        link1='https://youtube.com/search?q=best '+predict_new(image,model)+" in cricket"
        link2='https://youtube.com/search?q=top '+predict_new(image,model)+" by Sachin in cricket"
        link3='https://youtube.com/search?q=how to play '+predict_new(image,model)+" in cricket"
        link4='https://youtube.com/search?q=top '+predict_new(image,model)+" by Virat Kohli in cricket"


        if st.button('Best '+predict_new(image,model)+' in cricket'):
            link_youtube(link1)

        if st.button('Top '+predict_new(image,model)+' by sachin in cricket'):
            link_youtube(link2)

        if st.button('How to play '+predict_new(image,model)+' in cricket'):
            link_youtube(link3)

        if st.button('Top '+predict_new(image,model)+' by Virat Kohli in cricket'):
            link_youtube(link4)



    else:
        return("No cricketing shots detected")
    



if __name__== '__main__':
    main()
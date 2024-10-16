import streamlit as st
#from streamlit_image_zoom import image_zoom
import os
import cv2
from PIL import Image
import io
import psutil
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
#from sklearn.metrics import classification_report
from collections import Counter
import time
import pdf2image
from pdf2image import convert_from_bytes
from prediction_resnet import process_image

current_script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_script_directory)

parent_dir = os.path.abspath(os.pardir)
image_path = os.path.abspath(os.path.join(parent_dir, 'data', 'Images_bb'))

@st.cache_data
def load_image(imageName):
    current_dir = os.getcwd()
    image_path = os.path.join(current_dir, 'figures', imageName)
    image = Image.open(image_path)
    return image

@st.cache_data
def remove_uploaded_image(image_path):
    if os.path.exists(image_path):
        os.remove(image_path)

st.sidebar.title("Table of contents")
pages = [#"Project Introduction", "Data Exploration", "Feature Engineering", "Model Training", 
         "Model Optimization and Evaluation", "Model Inference" 
         ]
page = st.sidebar.radio("Go to", pages)

@st.cache_data
def local_css(file_name):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, file_name)
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if page == pages[5]:
    #st.write('# Detection and Classification of Defects on Printed Circuit Boards (PCBs)')
    st.html("<h1 style='text-align: center'>Detection and Classification of Defects on Printed Circuit Boards (PCBs) </hr> with Machine Learning</h1>")
    local_css("expander_bold.css")
    with st.expander("Introduction", expanded=False):
        st.write("- This project explores various machine learning methodologies for detecting and classifying defects on PCBs, using advanced computer vision techniques")
        st.write("- PCBs are essential components in nearly all electronic devices.")
        st.write("- Ensuring their quality is critical, as defects can lead to device malfunctions or failures.")
        st.write("- Traditional manual inspection methods are time-consuming and error-prone")
        st.write("- This motivates the adoption of deep learning models such as VGG16, RES-UNET, and YOLOv5 for automated defect detection.")
        st.write("- Our main object was therefore to learn the architecture designing, training and deployment of a manually designed RES_UNET model.")

    image_1 = load_image('PCB-Final-Image.jpg')
    st.image(image_1, caption="Typical 2 layer PCB", use_column_width='auto')

    st.write("### Project Phases")
    with st.expander("1 - Data Exploration", expanded=False):
        st.write("- Explore the dataset to understand its structure, features, and potential pitfalls.")
        st.write("- Use data visualization to identify key insights and relevance.")
        st.write("- Ensure the quality of the images.")
       
    with st.expander("2 - Feature Engineering"):
        st.write("- Perform feature engineering to balance and augment the data.")
        st.write("- Create the masks (segmentation) and the target labels (classification) for training images.")
        st.write("- Ensure the target labels were one-hot-encoded as per requirement of classification.")
        st.write("- Implement a randomization strategy to neutralize the impact of baises and ensure model impartiality.")
        st.write("- Ensuring the dataset is ready for model training.")
    
    with st.expander("3 - Model Training"):
        st.write("- Develop various machine learning models to detect the anomalies (defects).")
    
    with st.expander("4 - Model Optimization and Evaluation"):
        st.write("- Refine our designed model to maximize accuracy and robustness.")
        st.write("- Rigorous evaluation of model performance.")
      
    st.write("""This PCB defect detection project has been a dedicated effort for us, blending rigorous data 
analysis and advanced feature engineering with the practical application of machine learning. """)
    st.write("""Our goal was to excel in our endeavor by utilizing all available tools to successfully identify 
and classify PCB defects.""")

    
elif page == pages[4]:
    st.write("# Data Exploration")
    st.write("- The image dataset used has over 10,000 synthetically generated images.")
    st.write("- The dataset is publicly available at: https://www.kaggle.com/datasets/akhatova/pcb-defects)")
    
    st.write("#### Sample image for a defective PCB:")
    image_2 = load_image('sample_pcb_open_circuit.jpg')
    st.image(image_2, caption="Sample of a defected PCB", width=500)

    with st.expander("The types of defects detected in this project:", expanded=False):
        image_3 = load_image('Defect_types.png')
        st.image(image_3, caption="Sample defects explored in this project", use_column_width='auto')

    options = {'Missing Hole':'missing_hole', 'Mouse Bite':'mouse_bite', 'Open Circuit':'open_circuit', 'Short':'short', 'Spur':'spur_', 'Spurious Copper':'spurious_copper'}
    st.markdown('### Sample images with defects')
    choice = st.selectbox('Select Defect', options.keys(), index=0, label_visibility='collapsed')
    if choice is not None:
        img_pool_choice = [os.path.join(image_path, filename) for filename in os.listdir(image_path) if options[choice] in filename]
        rnd_2 = np.random.choice(range(6), 2, replace=False)
        fig = plt.figure(figsize=(15, 30))
        for i, j in enumerate(rnd_2):
            img = cv2.imread(img_pool_choice[j], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, 2, i + 1)
            plt.axis('off')
            plt.title(f"{choice} {i + 1}", fontsize=20)
            plt.imshow(img)
        st.pyplot(fig)
    st.write("It can be seen that each image can have more than one defect, although in each image there is only have one type of defect.")
    local_css('expander_regular.css')
    with st.expander(f"Dimensions of dataset images: {image_2.size}"):
        st.write("⭢ Those images are too large to handle without any pre-processing.")
		
		
    st.write("### Is there a possibilty to minimize the features?")
    with st.expander('View superimposed dataset images', expanded=False):
        image_4a = load_image("superimposed_image.png")
        st.image(image_4a, caption='All the training images superimposed together', width=500)
        st.write("- After inspecting the different images, there doesnt seem to be a possibility of reducing features, as this will result in essential features being lost.")
        st.write("- All the various nuances of the image need to be preserved for an accurate and robust anomaly detection model.")

elif page == pages[2]:
    st.write("# Feature Engineering")
    local_css('expander_medium.css')
    st.write("### 1. Visualization of the dataset")
    with st.expander("How balanced is our dataset?"):
        image_5 = load_image('data_balance.png')
        st.image(image_5, caption="Defect distribution", use_column_width='auto')
        st.write(" **Observations**")  
        st.write("- The dataset is relatively balanced overall")
        st.write("- Still, this is the visualization **before** doing feature engineering to ensure model robustness")
        st.write("- For model training, a randomly generated balanced dataset is used after performing the next preprocessing steps")

    st.write("### 2. Data Preprocessing")
    st.write("- Data preprocessing plays a crucial role in constructing effective Machine Learning models") 
    st.write("- The quality of prediction results is closely tied to the thoroughness of data preprocessing")
    st.write("- Our image preprocessing pipeline involved several key steps:")
    with st.expander("Dimension Handling"):
        st.write("- Reducing the image dimensions initially from RGB to Grayscale")
        image_6 = load_image('rgb vs grayscale.png')
        st.image(image_6, caption="Colored vs. Grayscale image", use_column_width='auto')
        
        st.write("- Cropping the the image to 100 x 100 grayscale images")
        image_6_1 = load_image('image_cropping.png')
        st.image(image_6_1, caption='Image Cropping', use_column_width='auto')
        st.write("- Some defects can be cut into two parts during that process")
        st.write("- That would influence the model training")

    with st.expander("Mask and Target Label"):
        st.write("- The dataset comes along with an xml file annotating for each image the bounding box of each defect and its type")
        st.write("- From the xml, for each image the pixel mask could be created with the same shape as the original image")

        image_6_2 = load_image('defect_img_vs_pm.png')
        st.image(image_6_2, caption="Original Image vs. Pixel Mask", use_column_width='auto')

        image_8 = load_image('croppedimagewithmask.png')
        st.image(image_8, caption="Cropping image and mask to 100x100 dimension", use_column_width='auto')

        st.write("- Those Pixel Masks will will become one of two labels for our model")
        st.write("- The second label will be the defect type")

    with st.expander("Augmentation"):
        st.write("- It was considered to utilize some ready-to-use image augmentation solutions like Image-Data-Generator or Albumentations")
        st.write("- For several reasons those have not been adequate for our dataset and our goal")
        st.write("- Hence manual augmentations were implemented:")

        image_9 = load_image('manual augmentation.png')
        st.image(image_9, caption="Manually implemented augmentations", use_column_width='auto')

    with st.expander("Separation of Multiple Defects"):
        st.write("- For model training, multiple defects in one cropped image were separated by image duplication")
        new_width = 400
        image_10 = load_image('separated_img1.png')
        image_10 = image_10.resize((new_width, int((new_width / image_10.width) * image_10.height)))
        st.image(image_10, use_column_width='auto')
        image_12 = load_image('separated_img3.png')
        image_12 = image_12.resize((new_width, int((new_width / image_12.width) * image_12.height)))
        st.image(image_12, caption="Manually implemented defect separation", use_column_width='auto')
    
elif page == pages[3]:
    local_css('expander_medium.css')
    st.write("# Model Training")
    st.write("For the development and implementation of the machine learning model, it went through many design iterations to finally decide on the RES-UNET model scheme.")
    with st.expander("1 - RES-UNET"):
        #image_13 = pdf2image.convert_from_bytes("pcb-resunet-model.pdf")
        pdf_path = os.path.join(current_script_directory , 'figures', 'pcb-resunet-model.pdf')
        pdf_bytes = open(pdf_path, "rb").read()
        images = convert_from_bytes(pdf_bytes)
        image_13 = images[0]
        #image_13 = image_zoom(image_13, size=700, zoom_factor=2.5)
        #image_13 = load_image('RESUNET_architecture.png')
        st.image(image_13, caption="RES-UNET model with Segmentation and Classification outputs", use_column_width='auto')

    local_css('expander_regular.css')
    with st.expander("2 - VGG16"):
        st.write("""The VGG16 model is a Convolutional Neural Network architecture that has been widely used for image classification tasks.""")
        st.write(" **Observations**:")
        st.write("- The resizing of images for VGG16 to 244,244 RGB dimensions causes alot of the features of the defects to be distorted")
        st.write("- Unfortunately this model was unable to present reasonably good output results for our segmentation task")
        st.write("- Hence it was decided to drop any further training on this pre-trained model")

    with st.expander("3 - YOLOv5"):
        st.write("In addition to designing and developing our RES_UNET model for training, the YOLOv5 object detection model developed by Ultralytics was also successfully implementedon the PCB datase.")
        st.write("This pre-trained model can be utilized for both segmentation and classification, providing us with the opportunity to compare the results of our model with this pretrained and well-established design.")
        st.write("For more details on model architecture please go to:")
        st.write("(https://docs.ultralytics.com/yolov5/tutorials/architecture_description/)")

elif page == pages[0]:
    st.write("# Model Optimization and Evaluation")
    st.write("""The performance of the models was evaluated using accuracy, MeanIoU, precision, recall, 
             and F1-score metrics, ultimately achieving a classification accuracy of 95%""")
    st.write("""The classification report and the confusion matrix on the validation set show that precision 
             and recall for each defect class vary but they are generally stable.""")
    
    image_14 = load_image('RESUNET_classification_report_v240618_1.png')
    st.image(image_14, caption="Metrics for classification output", 
             use_column_width='auto')
    
    image_15 = load_image('RESUNET_confusion_matrix_v240618_1.png')
    st.image(image_15, caption="Confusion Matrix for classification output", 
             width=500)
    
    st.write("##### RES-NET model Results")
    st.write("""The figure below illustrates that the location of the defects or the pixel matrix is predicted 
             quite precisely as Segmentation output. For classification output, the real and predicted classes 
             are shown along with confidence values for the label prediction.""")

    image_16 = load_image('RESUNET_prediction_examples_b_v240618_1.png')
    st.image(image_16, caption="Validation Results", 
             use_column_width='auto')

    st.write("""Finally a look at the prediction results along side the original images to demonstrate a high 
             degree of accuracy and precision, highlighting the effectiveness and potential of this machine 
             learning model.""")  
    image_17 = load_image('results1_resunet.png')
    st.image(image_17, use_column_width='auto')
    image_18 = load_image('results2_resunet.png')
    st.image(image_18, use_column_width='auto')
    image_19 = load_image('results3_resunet.png')
    st.image(image_19, caption="Final prediction Results for RES-UNET model", 
             use_column_width='auto')
         

#model inference
elif page == pages[1]:
    st.write("# Model Inference")
    st.write("""Now that the efficient model is trained and validated, let's determine if it can predict the defects 
             accurately. The model inference can be run on pre-loaded test images or the user can upload thier own 
	     images for testing.""")
	#i am working on uploading various images for testing
    
    test_options = ["test_image_1.jpg", "test_image_2.jpg", "test_image_3.jpg",
                    "test_image_4.jpg", "test_image_5.jpg", "test_image_6.jpg",
                    "test_image_7.jpg", "test_image_8.jpg", "test_image_9.jpg"]

    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
	    
    tab1, tab2 = st.tabs(["Initialize Test Image", "Defect Detection"])
    uploaded_selection = None
    test_selection = None
    
    with tab1:
        # select test image 
        test_source = st.radio("Choose test source", 
                               ('Use a pre-loaded test image', 'Upload a custom test image'))

        if test_source == 'Use a pre-loaded test image':
            test_option = st.selectbox('Choose from different test samples', 
                                       options=(1, 2, 3, 4, 5, 6, 7, 8, 9)
            )
            test_selection = os.path.join('test_images', test_options[test_option-1])
            disp_test_selection = load_image(test_selection)
            st.image(disp_test_selection, caption="Selected Test Image", width=700)
        
        else:
            uploaded_selection = st.file_uploader("Upload a PCB image",
                                                  type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
            if "unique_filename" in st.session_state and st.session_state.unique_filename is not None:
                uploaded_image_path = os.path.join(current_script_directory, 'figures', st.session_state.unique_filename)
                remove_uploaded_image(uploaded_image_path)
                st.session_state.unique_filename = None

            if uploaded_selection is not None:
                current_time = int(time.time())
                unique_filename = f"{current_time}_{uploaded_selection.name}"
                image_path = os.path.join(current_script_directory, 'figures', unique_filename)
                with open(image_path, 'wb') as f:
                    f.write(uploaded_selection.read())
                    
                st.session_state.unique_filename = unique_filename
                st.session_state.uploaded_images.append(unique_filename)

                if len(st.session_state.uploaded_images) > 1:
                    old_filename = st.session_state.uploaded_images.pop(0)
                    old_image_path = os.path.join(current_script_directory, 'figures', old_filename)
                    remove_uploaded_image(old_image_path)
		
            #if uploaded_selection is not None:
                test_options = [unique_filename]
                test_selection = test_options[0]
                disp_test_selection = load_image(test_selection)
                st.image(disp_test_selection, caption="Selected Test Image", width=700)

        # on_click of predict button
        if st.button('View mask prediction', key='show_mask'):
            with st.spinner('Wait for a few seconds...'):
                time.sleep(2)
            # run image processing pipeline
            image, num_defects = process_image(1, test_img=('./figures/'+test_selection))
            st.image(image,
                    caption=None, width=700
            )
            st.write("Number of masks detected: ", num_defects)
            st.success('Finished mask prediction.')
   
    with tab2:
        st.write("Classifying and drawing bounding boxes around defects on the original image.")
        if (test_selection is not None):
            disp_test_selection = load_image(test_selection)
            st.image(disp_test_selection, caption="Selected Test Image", width=700)
            # predicttion on_click of button
            if st.button('Show defects', key='show_defects_model'):
                with st.spinner('Wait for a few seconds...'):
                    time.sleep(2)
                # run image processing pipeline
                image, num_defects = process_image(2, test_img=('./figures/'+test_selection))
                st.image(image,
                        caption=None, width=700
                )
                st.write("Number of defects detected: ", num_defects)
                st.success('Finished defects localisation and classification.')


        

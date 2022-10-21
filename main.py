from PIL import Image
import torch
import cv2
import numpy as np
from torchvision import datasets, transforms
# from model.ResNet import resnet18
import os
import matplotlib.pyplot as plt
#import toml
import streamlit as st
from scipy import interpolate
from GradCam_v3 import GradCam
import math
import multiprocessing
from itertools import repeat
import torch.nn as nn


st.set_page_config(

    page_title="MA Demo",
)
st.title('Membrane Analysis Demo')
STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

primaryColor = "#000000"
s = f"""
<style>
div.stButton > button:first-child {{ border: 5px solid {primaryColor}; background-color: #D0F0C0; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)

def main():

    st.sidebar.title("What to do")
    app_mode = st.sidebar.radio("Choose the app mode",
                                    ["Project Overview", "Classification", "FAQ"])

    if app_mode == "Project Overview":
        #edge_img = Image.open('./edge.jpg')
        #full_image = Image.open('./complete.jpg')
        #header1=st.header("Importance")
        #placeholder1 = st.image(edge_img, width=900, caption="Edge Location: Pixel Level Precision is required")

        header2=st.header("Project Overview")
        # placeholder2=st.image(full_image, width=900, caption="(a) Classification module for identification of normal or "
        #                                         "abnormal edge and welded or non-welded edge.  (b) Edge detection "
        #                                         "module which is a modified UNET architecture. 3×3 convolution in the "
        #                                         "second last layer is replaced with 2×2 max pooling layer to get 1D output. "
        #                                         "The final output of (b) is 1×W, i.e., 1×43 where the boundary between0s"
        #                                         " and 1s indicates an edge.")
        #header3 = st.header("Labeling Demo")

        #video_file = open('./video.mp4', 'rb')
        #video_bytes = video_file.read()
        #placeholder3=st.video(video_bytes)


    # elif app_mode == "Edge Detection":
    #
    #     st.subheader('Welcome to Edge Detection Module')
    #     edge_detection()

    elif app_mode == "Classification":

        st.subheader('Welcome to Classification Module')
        classification()

    elif app_mode == "FAQ":

        st.subheader("Frequently Asked Questions")
        faq()

def faq():
    e1 = st.beta_expander("Why are the images so bad?")
    e2 = st.beta_expander("Can you use a better camera?")
    e3 = st.beta_expander("Can you describe the procedure?")
    e4 = st.beta_expander("Why are you using AI? Isn't there a cheaper option?")
    e5 = st.beta_expander("Can we enhance the image, that the worker can do a better/easier job?")
    e6 = st.beta_expander("How much do we save?")


# def edge_detection():
#     st.sidebar.markdown("# Data")
#     data_type = st.sidebar.radio("Do you want to upload your own data or select images from given database?",
#                                        ["Own Data", "Database"])
#     if data_type=="Own Data":
#         file_selector_O()
#     if data_type=="Database":
#         file_selector_DB()

def classification():

    st.sidebar.markdown("# Classification Type")
    data_type = st.sidebar.radio(
        "Do you want to classify defect/not-defect image dataset or easy/difficult images?",
                                       ["Defect/Not-Defect"])
                                           #, "Easy/Difficult"])

    if data_type == "Defect/Not-Defect":

        defect_notdefect()
    # if data_type == "Easy/Difficult":
    #
    #     easy_difficult()

# def easy_difficult():
#     st.sidebar.markdown("# Data")
#     data_type = st.sidebar.radio("Do you want to upload your own data or select images from given database?",
#                                  ["Own Data", "Database"])
#     if data_type == "Own Data":
#         easy_difficult_O()
#     if data_type == "Database":
#         easy_difficult_DB()

# def easy_difficult_DB():
#     folder_path = "./Datasets/easy_difficult"
#     filenames = os.listdir(folder_path)
#     filenames1 = ['None']
#     list = filenames1 + filenames
#
#     selected_filename = st.sidebar.selectbox('Select a file', list)
#
#     if selected_filename != 'None':
#         file_path = os.path.join(folder_path, selected_filename)
#         pil_img = Image.open(file_path)
#
#         img_nd = preprocess(pil_img)
#         st.image(img_nd, caption="Full Image")
#
#         transform = transforms.Compose([
#             transforms.CenterCrop(87),
#             transforms.ToTensor()
#         ])
#         pil_img = Image.fromarray(np.uint8(img_nd)).convert('RGB')
#         data = transform(pil_img)
#         imgs = torch.unsqueeze(data, 0)
#         pred = classification_evaluation('./checkpoint/classification/nonweld_weld/epoch290_model.pth.tar', imgs)
#         ROI = img_nd[140:227, 287:330]
#         col1, col2, col3 = st.beta_columns(3)
#         col2.image(ROI, width=150, caption="ROI")
#         col1, col2, col3 = st.beta_columns(3)
#         classify = col2.button('Classify')
#         if classify:
#
#
#             if pred == 0:
#                 title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Edge is difficult to locate</p>'
#
#                 st.markdown(title, unsafe_allow_html=True)
#             else:
#                 title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Edge is easy to locate</p>'
#
#                 st.markdown(title, unsafe_allow_html=True)


# def easy_difficult_O():
#
#     st.info("Try as many times as you want!")
#     fileTypes = ["jpeg", "png", "jpg", "bmp"]
#     st.markdown(STYLE, unsafe_allow_html=True)
#     file = st.file_uploader("Upload image", type=fileTypes)
#
#     show_file = st.empty()
#     if not file:
#         show_file.info("Please upload an image of type: " + ", ".join(["jpeg", "png", "jpg", "bmp"]))
#         return
#     # if isinstance(file, BytesIO):
#     #   show_file.image(file, caption="Full Image")
#
#     pil_img = Image.open(file)
#
#     img_nd = preprocess(pil_img)
#     st.image(img_nd, caption="Full Image")
#     transform = transforms.Compose([
#         transforms.CenterCrop(87),
#         transforms.ToTensor()
#     ])
#     pil_img = Image.fromarray(np.uint8(img_nd)).convert('RGB')
#     data = transform(pil_img)
#     imgs = torch.unsqueeze(data, 0)
#     pred = classification_evaluation('./checkpoint/classification/difficult_easy/epoch500_model.pth.tar', imgs)
#     ROI = img_nd[140:227, 287:330]
#     col1, col2, col3 = st.beta_columns(3)
#     col2.image(ROI, width=150, caption="ROI")
#     classify = col2.button('Classify')
#     if classify:
#
#         if pred == 0:
#             title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Edge is difficult to locate</p>'
#
#             st.markdown(title, unsafe_allow_html=True)
#         else:
#             title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Edge is easy to locate</p>'
#
#             st.markdown(title, unsafe_allow_html=True)
#
#         file.close()
#         st.info("To preceed further please clear the data by pressing cross under the browse button")


def defect_notdefect():
    st.sidebar.markdown("# Data")
    data_type = st.sidebar.radio("Do you want to upload your own data or select images from given database?",
                                 ["Own Data", "Database"])
    if data_type == "Own Data":
        defect_notdefect_O()
    if data_type == "Database":
        defect_notdefect_DB()

def defect_notdefect_DB():

    folder_path = "./Datasets/defect_notdefect"
    filenames = os.listdir(folder_path)
    filenames1 = ['None']
    list = filenames1 + filenames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('model/DefectClassifier.pth')
    # checkpoint = './checkpoint/defect_classifier_densenetAugDataset1.pth.tar'
    checkpoint = './checkpoint/defect_classifier_densenet_22_12_2.pth.tar'

    selected_filename = st.sidebar.selectbox('Select a file', list)

    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ffffff;
        color: #000000;
        border-color: #ffffff;
    }
    </style>""", unsafe_allow_html=True)

    if selected_filename != 'None':
        file_path = os.path.join(folder_path, selected_filename)

        ext = os.path.splitext(file_path)[-1].lower()

        # Now we can simply use == to check for equality, no need for wildcards.
        if ext == ".mp4":

            if f'{folder_path}/temp.mp4':
                os.remove(f'{folder_path}/temp.mp4')

            os.system(f'ffmpeg -i {file_path} -vcodec libx264 {folder_path}/temp.mp4')

            video_file = open(f'{folder_path}/temp.mp4', 'rb')
            #video_bytes = video_file.read()

            st.video(video_file)

            #placeholder2 = st.video(file_path)


            st.markdown(
                "<h1 style='text-align: center; color:Black; font-size: 20px;'>Video With Defect Classification</h1>",
                unsafe_allow_html=True)

        else:
            pil_img = Image.open(file_path)

            # img_nd = preprocess(pil_img)
            img_nd = pil_img

            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])
            ])

            classify_obj = classification_evaluation(mdl=model, trans=transform, checkpoint_path=checkpoint,
                                             device=device, img=img_nd)

            st.image(img_nd, caption="Full Image", width=800)
            st.text("")
            _, col1, col2, col3 = st.columns([1, 1, 1, 1])
            classify = col1.button('Classify', key=101)
            gram = col2.button('AIX 1', key=102)
            rise = col3.button('AIX 2', key=103)

            if classify:

                if classify_obj() == 0:
                    st.text("")
                    title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">It is a defective membrane</p>'

                    st.markdown(title, unsafe_allow_html=True)

                elif classify_obj() == 1:
                    st.text("")
                    title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">It is a good membrane</p>'

                    st.markdown(title, unsafe_allow_html=True)


            if gram:
                show_raw = False

                classify_obj.GradCam()
                # gc_img = Image.open(f'/home/sarmad/PycharmProjects/MA_Demo/temp.jpg')
                gc_img = Image.open(f'./temp.jpg')

                gc_img = preprocess(gc_img)

                st.text("")
                st.text("")


                #v_spacer(height=3, sb=True)
                st.image(gc_img, caption="AIX 1", width=800)


            if rise:
                classify_obj.Rise()
                # gc_img = Image.open(f'/home/sarmad/PycharmProjects/MA_Demo/temp.jpg')
                gc_img = Image.open(f'./temp.jpg')

                gc_img = preprocess(gc_img)
                st.text("")
                st.text("")

                v_spacer(height=3, sb=True)
                st.image(gc_img, caption="AIX 2", width=800)

def defect_notdefect_O():

    st.info("Try as many times as you want!")
    fileTypes = ["jpeg", "png", "jpg", "bmp"]
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload image", type=fileTypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('model/DefectClassifier.pth')
    # checkpoint = './checkpoint/defect_classifier_densenetAugDataset1.pth.tar'
    checkpoint = './checkpoint/defect_classifier_densenet_22_12_2.pth.tar'

    m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #ffffff;
            color: #000000;
            border-color: #ffffff;
        }
        </style>""", unsafe_allow_html=True)

    show_file = st.empty()
    if not file:
        show_file.info("Please upload an image of type: " + ", ".join(["jpeg", "png", "jpg", "bmp"]))
        return
    #if isinstance(file, BytesIO):
     #   show_file.image(file, caption="Full Image")

    pil_img = Image.open(file)

    img_nd = preprocess(pil_img)
    st.image(img_nd, caption="Full Image", width=800)
    st.text("")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    classify_obj = classification_evaluation(mdl=model, trans=transform, checkpoint_path=checkpoint,
                                             device=device, img=img_nd)
    #ROI = img_nd[140:227, 287:330]
    _, col1, col2, col3 = st.columns([1, 1, 1, 1])
    classify = col1.button('Classify', key=101)
    gram = col2.button('AIX 1', key=102)
    rise = col3.button('AIX 2', key=103)

    if classify:

        if classify_obj() == 0:
            st.text("")
            title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">It is a defective membrane</p>'

            st.markdown(title, unsafe_allow_html=True)

        elif classify_obj() == 1:
            st.text("")
            title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">It is a good membrane</p>'

            st.markdown(title, unsafe_allow_html=True)

    if gram:
        show_raw = False

        classify_obj.GradCam()
        # gc_img = Image.open(f'/home/sarmad/PycharmProjects/MA_Demo/temp.jpg')
        gc_img = Image.open(f'./temp.jpg')

        gc_img = preprocess(gc_img)

        st.text("")
        st.text("")

        # v_spacer(height=3, sb=True)
        st.image(gc_img, caption="AIX 1", width=800)

    if rise:
        classify_obj.Rise()
        # gc_img = Image.open(f'/home/sarmad/PycharmProjects/MA_Demo/temp.jpg')
        gc_img = Image.open(f'./temp.jpg')

        gc_img = preprocess(gc_img)
        st.text("")
        st.text("")

        v_spacer(height=3, sb=True)
        st.image(gc_img, caption="AIX 2", width=800)


def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')


def detect_circles(img_name, size, dim=None, HOUGH=True):

    # given image, center image around circle and crop to given size.
    # appends when necessary.

    blur_kernel = (15, 15)
    param1 = 100
    param2 = 60
    minRadius = 10
    maxRadius = 600

    # Read image.
    img = img_name

    # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    if HOUGH == True:
        # img = cv2.cvtColor(img, cv2.CV_BGR2RGB)
        img = img[:,:,::-1].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, blur_kernel)

        # Apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blurred,
                                            cv2.HOUGH_GRADIENT, 1, 1, param1=param1,
                                            param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        # blank = np.zeros([gray.shape[0], gray.shape[1]], dtype=np.float32)
        # Draw circles that are detected.
        if detected_circles is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))

            for pt in detected_circles[0, :]:
                # print("Circle Detected")
                a, b, r = pt[0], pt[1], pt[2]

                # cv2.circle(blank, (np.int32(a), np.int32(b)), np.int32(1.1 * r - int(0 / 4)), color=(255), thickness=-1)
                # img[blank == 0, :] = 0

                # Draw the circumference of the circle.
                # cv2.circle(img, (a, b), r, 255, 5)

                # Draw a small circle (of radius 1) to show the center.
                # cv2.circle(img, (a, b), 1, 255, 3)

                # print(a, b)
                # print(gray.shape)
                # bottom
                if b + size > gray.shape[0]:
                    # print("Bottom")
                    img = cv2.copyMakeBorder(img, 0, b + size - gray.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                # top
                if b - size < 0:
                    # print("Top")
                    img = cv2.copyMakeBorder(img, size - b, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    b += (size - b)

                # right
                if a + size > gray.shape[1]:
                    # print("Right")
                    img = cv2.copyMakeBorder(img, 0, 0, a + size - gray.shape[1], 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                # left
                if a - size < 0:
                    # print("Left")
                    img = cv2.copyMakeBorder(img, 0, 0, 0, size - a, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    a += (size - a)

                img = img[b - size:b + size, a - size:a + size]

                return img

    x_mid, y_mid = int(img.shape[0]/2), int(img.shape[1]/2)

    if x_mid + size > img.shape[0]:
        # print("Bottom")
        img = cv2.copyMakeBorder(img, 0, x_mid + size - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # top
    if x_mid - size < 0:
        # print("Top")
        img = cv2.copyMakeBorder(img, size - x_mid, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x_mid += (size - x_mid)

    # right
    if y_mid + size > img.shape[1]:
        # print("Right")
        img = cv2.copyMakeBorder(img, 0, 0, y_mid + size - img.shape[1], 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # left
    if y_mid - size < 0:
        # print("Left")
        img = cv2.copyMakeBorder(img, 0, 0, 0, size - y_mid, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        y_mid += (size - y_mid)

    img = img[x_mid - size:x_mid + size, y_mid - size:y_mid + size]

    return img


def create_mask(size=80, up_size=200, new_size=224):

    mask = np.zeros([3, up_size, up_size])
    a = np.random.randint(low=0, high=2, size=[size, size])

    x = np.array(range(size))
    y = np.array(range(size))

    f = interpolate.interp2d(x, y, a, kind='cubic')

    xnew = np.linspace(0, 2, up_size)
    ynew = np.linspace(0, 2, up_size)
    znew = f(xnew, ynew)

    mask[0] = znew
    mask[1] = znew
    mask[2] = znew

    return mask


# def classification_evaluation(checkpoint_path, data, device):
#
#
#     model = torch.load('model/DefectClassifier.pth').to(device)
#     model.eval()
#
#     model.load_state_dict(torch.load(checkpoint_path))
#     output = model(data)
#
#     pred = torch.argmax(output)
#
#     return pred.item()
def sub_rise(j, model, crop, normalize, centered_img_PIL, pred):
    #print('j')
    mask = create_mask(size=50, up_size=600)
    mask = torch.FloatTensor(mask)
    mask = crop(mask.unsqueeze(0))
    inp = torch.mul(mask, centered_img_PIL)
    logps = model(normalize(inp))

    ps = torch.exp(logps[0][pred])
    print(j)
    return torch.mul(ps, mask)


def sub_GradCAM(img, c, features_fn, classifier_fn):
    feats = features_fn(img)
    # print(feats.shape)
    _, N, H, W = feats.size()
    out = torch.exp(classifier_fn(feats))
    # print(out)
    c_score = out[0, c]
    # print(c_score)
    grads = torch.autograd.grad(c_score, feats)
    # print(grads[0].shape)
    w = grads[0][0].mean(-1).mean(-1)
    # print(w.shape)
    sal = torch.matmul(w, feats.view(N, H*W))
    # print(sal.shape)
    sal = sal.view(H, W).cpu().detach().numpy()
    # print(sal.shape)
    sal = np.maximum(sal, 0)
    # print(sal.shape)
    return sal


class Flatten(nn.Module):
    """One layer module that flattens its input."""
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


class classification_evaluation:

    def __init__(self, mdl, trans, checkpoint_path, device, img):

        self.model = mdl.to(device)
        self.raw_img = img
        #self.transform = trans
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.image = trans(Image.fromarray(np.uint8(detect_circles(img_name=img, size=600))).convert('RGB')).unsqueeze(0)
        self.pred = None

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def _crop_img(self, img):
        pass

    def __call__(self):

        output = self.model(self.image)
        self.pred = torch.argmax(output)

        return self.pred.item()

    def GradCam(self):

        for param in self.model.parameters():
            param.requires_grad = True

        features_fn = self.model.features
        classifier_fn = nn.Sequential(*([nn.AvgPool2d(7, 1), Flatten()] + [self.model.classifier]))

        output = self.model(self.image)
        self.pred = torch.argmax(output)

        pred_label = 'Defect' if self.pred == 0 else 'Good'
        #plt.figure(figsize=(20, 20))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        ax = [ax1, ax2]

        for ij in range(2):
            #plt.subplot(1, 2, ij + 1)
            sal = sub_GradCAM(self.image, int(ij), features_fn, classifier_fn)
            #img = cv2.imread(img_path)
            sal = Image.fromarray(sal)
            sal = sal.resize((self.raw_img.shape[1], self.raw_img.shape[0]), resample=Image.LINEAR)

            ax[ij].axis('off')
            ax[ij].imshow(self.raw_img)
            ax[ij].imshow(np.array(sal), alpha=0.5)
            #     plt.contourf(np.array(sal)/np.array(sal).max(),
            #                  levels=np.append(np.linspace(0.1, 0.8, 4), np.linspace(0.81, 1, 5)), alpha=0.5)
            # plt.colorbar(cb)

            title = 'Defect' if ij == 0 else 'Good'
            ax[ij].set_title(f'Logit: {title}', fontsize=14)

        fig.suptitle(f'Classification: {pred_label}', fontsize=16)
        # plt.xlabel(f'{num}: Defect-{filename}')
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #fig.subplots_adjust(wspace=0.4, hspace=None)

        # plt.savefig(f'/home/sarmad/PycharmProjects/MA_Demo/temp.jpg', bbox_inches='tight')
        plt.savefig(f'./temp.jpg', bbox_inches='tight')
        plt.close()

    def GradCam2(self):

        for param in self.model.parameters():
            param.requires_grad = True

        output = self.model(self.image)
        self.pred = torch.argmax(output)

        pred_label = 'Defect' if self.pred == 0 else 'NotDefect'

        nlayers = len(self.model.features._modules.items()) - 1

        fig, ax = plt.subplots(math.ceil(nlayers / 4), 4, figsize=(20, 10))
        ax = ax.flatten()

        for i, layer in enumerate(self.model.features._modules.items()):
            # print(layer[0])
            grad_cam = GradCam(self.model, target_layer=layer[0])
            cam = grad_cam.generate_cam(self.image, self.pred.item())
            ax[i].imshow(cam)
        plt.suptitle(f'Classification: {pred_label} - Logit: {pred_label}', fontsize=22)
        # plt.savefig(f'/home/sarmad/PycharmProjects/MA_Demo/temp.jpg', bbox_inches='tight')
        plt.savefig(f'./temp.jpg', bbox_inches='tight')
        plt.close()

    def Rise(self):
        no_of_masks = 2000
        crop = transforms.RandomCrop(224)
        normalize = transforms.Normalize([0.5, 0.5, 0.5],
                                         [0.5, 0.5, 0.5])

        trans2 = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor()
                                     ])

        centered_img_cv = detect_circles(img_name=self.raw_img, size=600)
        centered_img_PIL = trans2(Image.fromarray(np.uint8(detect_circles(img_name=self.raw_img, size=600))))

        output = self.model(self.image)
        self.pred = torch.argmax(output)

        pred_label = 'Defect' if self.pred == 0 else 'Good'

        visual0 = torch.zeros_like(self.image.squeeze())
        visual1 = torch.zeros_like(self.image.squeeze())

        for i in range(no_of_masks):
            mask = create_mask(size=50, up_size=600)
            mask = torch.FloatTensor(mask)
            mask = crop(mask.unsqueeze(0))
            inp = torch.mul(mask, centered_img_PIL)
            logps = self.model(normalize(inp))

            ps0 = torch.exp(logps[0][0])
            ps1 = torch.exp(logps[0][1])

            visual0 += torch.mul(ps0, mask.squeeze())
            visual1 += torch.mul(ps1, mask.squeeze())

        heatmap0 = torch.mean(visual0.permute(1, 2, 0), dim=2) / torch.max(
            torch.max(torch.mean(visual0.permute(1, 2, 0), dim=2)))

        heatmap1 = torch.mean(visual1.permute(1, 2, 0), dim=2) / torch.max(
            torch.max(torch.mean(visual1.permute(1, 2, 0), dim=2)))

        heatmap0 = cv2.resize(np.float32(heatmap0), (self.raw_img.shape[1], self.raw_img.shape[0]))
        heatmap1 = cv2.resize(np.float32(heatmap1), (self.raw_img.shape[1], self.raw_img.shape[0]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        ax1.axis('off')
        ax1.imshow(self.raw_img)
        #ax1.contourf(heatmap0, levels=[0.8, 0.9, 0.99, 0.999, 1], alpha=0.5)
        ax1.imshow(heatmap0, alpha=0.5)
        ax1.set_title(f'Logit: Defect', fontsize=14)


        ax2.axis('off')
        ax2.imshow(self.raw_img)
        #ax2.contourf(heatmap1, levels=[0.8, 0.9, 0.99, 0.999, 1], alpha=0.5)
        ax2.imshow(heatmap1, alpha=0.5)
        ax2.set_title(f'Logit: Good', fontsize=14)

        fig.suptitle(f'Classification: {pred_label}', fontsize=16)
        #plt.subplots_adjust(wspace=None)
        # plt.savefig(f'/home/sarmad/PycharmProjects/MA_Demo/temp.jpg', bbox_inches='tight')
        plt.savefig(f'./temp.jpg', bbox_inches='tight')
        plt.close()

def preprocess(pil_img):
    img_nd = np.array(pil_img)

    if len(img_nd.shape)<3:
        img_nd=cv2.cvtColor(img_nd, cv2.COLOR_GRAY2RGB)
    # img_nd=pil_img
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    return img_nd


def HWCtoCHM(img_nd):
    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans

def file_selector_O():


    st.info("Try as many times as you want!")
    fileTypes = ["jpeg", "png", "jpg","bmp"]
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload image", type=fileTypes)
    show_file = st.empty()
    if not file:
        show_file.info("Please upload an image of type: " + ", ".join(["jpeg", "png", "jpg","bmp"]))
        return
    #if isinstance(file, BytesIO):
     #   col1.show_file.image(file,caption="Full Image")

    pil_img = Image.open(file)
    img_nd=preprocess(pil_img)

    st.image(img_nd,caption="Full Image")
    ROI = img_nd[140:227, 287:330]

    ROI_CHW = HWCtoCHM(ROI)
    col1, col2, col3 = st.beta_columns(3)
    col2.image(ROI, width=200, caption="ROI")
    edge = col2.button("Detect Edge")

    if edge:


        edge_pixel=find_edge(ROI_CHW)

        image=img_nd.copy()
        cv2.line(image, (287 + edge_pixel, 80), (287 + edge_pixel, 300), (255,255,0), 1)
        cv2.putText(image, 'Prediction', (450, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255,255,0), thickness=2)
        cv2.rectangle(image, (287, 227), (330, 140), (0, 0, 255), 1)

        st.image(image, caption="Full Image with Edge")
        ROI_plot = ROI.copy()
        cv2.line(ROI_plot, (edge_pixel, 0), (edge_pixel, 87), (255, 0, 255), 1)

        col1, col2, col3 = st.beta_columns(3)

        col2.image(ROI_plot, width=200, caption="ROI with Edge")

        file.close()

        st.info("To preceed further please clear the data by pressing cross under the browse button")


def file_selector_DB():


    folder_path = "./Datasets/Edge_Detection"
    filenames = os.listdir(folder_path)
    filenames1 = ['None']
    list=filenames1+filenames

    selected_filename = st.sidebar.selectbox('Select a file', list)
    if selected_filename!='None':
        file_path = os.path.join(folder_path, selected_filename)
        img_name = selected_filename
        gt_edge = int(img_name[10:12])
        pil_img = Image.open(file_path)
        img_nd = preprocess(pil_img)
        ROI = img_nd[140:227, 287:330]

        ROI_CHW = HWCtoCHM(ROI)
        st.image(img_nd, caption="Full Image")
        col1, col2, col3 = st.beta_columns(3)
        col2.image(ROI, width=170, caption="ROI")
        edge= col2.button("Detect Edge")

        if edge:



            edge_pixel=find_edge(ROI_CHW)

            image=img_nd.copy()



            cv2.rectangle(image, (287, 227), (330, 140), (255, 0, 0), 1)
            cv2.line(image, (287 + gt_edge, 80), (287 + gt_edge, 300), (255, 0, 255), 1)
            cv2.putText(image, 'GT', (450, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 255), thickness=2)

            cv2.line(image, (287 + edge_pixel, 80), (287 + edge_pixel, 300), (255, 255, 0), 1)
            cv2.putText(image, 'Prediction', (450, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 0), thickness=2)
            st.image(image,caption="Full Image with Edge")


            ROI_plot=ROI.copy()
            cv2.line(ROI_plot, (gt_edge, 0), (gt_edge, 87), (255, 0, 255), 1)
            cv2.line(ROI_plot, (edge_pixel, 0), (edge_pixel, 87), (255, 255, 0), 1)



            col1, col2, col3 = st.beta_columns(3)

            col2.image(ROI_plot, width=170, caption="ROI with Edge")
            st.markdown(
                "<h1 style='text-align: center; color:Black; font-size: 20px;'> %d px difference b/w GT and Network</h1>" % (
                    abs(gt_edge - edge_pixel)), unsafe_allow_html=True)




def find_edge(test_img):

    model=UNet(n_channels=3, n_classes=1, bilinear=True)
    #torch.cuda.set_device(0)
    #model = model.cuda(0)

    checkpoint = torch.load('./checkpoint/detection/model_best.pth.tar',map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['state_dict'])
    dataset_test= torch.from_numpy(test_img).type(torch.FloatTensor)
    edge_pixel=validate(dataset_test, model)
    return edge_pixel


def validate(dataset_test, model):


    # switch to evaluate mode
    model.eval()


    with torch.no_grad():


        imgs = torch.unsqueeze(dataset_test,0)
        #imgs = imgs.cuda(0, non_blocking=True)
        # compute output
        with torch.no_grad():
            mask_pred = model(imgs)

        pred = torch.sigmoid(mask_pred)
        pred = (pred > 0.5).float()
        probs=pred
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )

        for i in range(len(probs)):

            probs=pred[i]
            probs = tf(probs.cpu())
            full_mask = probs.cpu().numpy()
            full_mask_ones = np.where(full_mask[0] == 1)
            full_mask_ones = full_mask_ones[1]
            edge_pixel = full_mask_ones[0]

    return edge_pixel



if __name__ == "__main__":

    main()

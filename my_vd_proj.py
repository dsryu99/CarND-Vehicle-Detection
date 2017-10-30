import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

MAX_FRAME = 5
frame_box = [[] for i in range(MAX_FRAME)]
frame_cur_idx = 0

stored_idx = -999
current_idx = 0
stored_labels = np.empty(())

dist_pickle = pickle.load(open("output_images/svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

#        print("bbox[0]:" + str(bbox[0]) + ", bbox[1]:"+ str(bbox[1]))
    # Return the image
    return img


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    # png (train) -> jpg (test) : the below is required,  jpg (train) -> jpg (test) : the below is not required
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, xstart:xstop, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bbox_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
            #if True:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                box = ((xbox_left + xstart, ytop_draw + ystart), (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart))
                bbox_list.append(box)
                #cv2.rectangle(draw_img, (xbox_left + xstart, ytop_draw + ystart),
                #              (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart), (0, 0, 255), 5)
    #plt.imshow(draw_img)
    #plt.show()
    return bbox_list

def process_image(image):
    draw_img = np.copy(image)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    bbox_all_list = []
    for ystart, ystop, xstart, xstop, scale in search_space:
        bbox_list = find_cars(image, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)
        if (len(bbox_list) > 0):
            #print("bbox_list:"+str(bbox_list))
            bbox_all_list.extend(bbox_list)

    global frame_cur_idx
    global frame_box
    global MAX_FRAME

    frame_box[frame_cur_idx] = np.copy(bbox_all_list)
    frame_cur_idx = (frame_cur_idx + 1) % MAX_FRAME
    all_list = []
    for i in range(MAX_FRAME):
        #print("frame_box:"+str(i)+":"+str(frame_box[i])+":"+str(type(frame_box[i])))
        all_list.extend(frame_box[i])

    global current_idx
    global stored_idx
    global stored_labels

    if (len(all_list) > 0):
        heat_m = add_heat(heat, all_list)
        heat_m = apply_threshold(heat_m, 15) # 20
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat_m, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
#        plt.imshow(labels[0], cmap='gray')
#        plt.show()

        if labels[1] > 0:
            stored_idx = current_idx
            stored_labels = np.copy(labels)
        else:
            if (stored_idx == current_idx - 1):
                stored_idx = current_idx
                labels = np.copy(stored_labels)

        draw_img = draw_labeled_bboxes(np.copy(image), labels)

    current_idx = current_idx + 1
    return draw_img

# (ystart, ystop, xstart, xstop, scale)
min_search = (400, 500, 450, 1050, 1.0)
mid1_search = (400, 550, 450, 1050, 1.3)
mid2_search = (390, 600, 350, 1100, 1.5)
mid3_search = (390, 600, 350, 1280, 1.8)
mid4_search = (390, 600, 300, 1280, 2.0)
mid5_search = (340, 670, 100, 1280, 4.0)
max_search = (230, 720, 100, 1280, 6.0)
search_space = [min_search, mid1_search, mid2_search, mid3_search, mid4_search, mid5_search, max_search]

'''
img = mpimg.imread('test_images/test4.jpg')
#img = mpimg.imread('bbox-example-image.jpg')
out_img = process_image(img)
cv2.imwrite("output_result.jpg", out_img)
plt.imshow(out_img)
plt.show()
quit()
'''

'''
#0 ~ 1251
for fno in range(931, 937): # 1033 ~ 1045  # 480 ~ 750  # 506 ~ 511, 660 ~ 730
    file_name = "frame"+str(fno)+".jpg"
    print(file_name)
    in_img = mpimg.imread('project_in_images/'+file_name)
    re_img = process_image(in_img)
#    cv2.imwrite('project_out_images/'+file_name, re_img)
    plt.imshow(re_img)
    plt.show()
quit()
'''

white_output = 'output_videos/project_video.mp4'
#clip1 = VideoFileClip("project_video.mp4").subclip(26,31)      # 1280 x 720
clip1 = VideoFileClip("project_video.mp4")      # 1280 x 720
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
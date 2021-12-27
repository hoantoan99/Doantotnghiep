from libs import *


# Hàm crop ảnh và vị trí cần crop
# Crop ảnh vị trí bị lỗi
def image_crop(image, alpha, alpha_x1, alpha_y1, alpha_x2, alpha_y2):
    # crop bảng mạch
    # thực hiện vẽ contour, max contour chính là bảng mạch
    # sử dụng hàm approxPolyDP để tìm 4 góc của bảng mạch
    # hệ số alpha càng nhỏ thì số lượng điểm tìm được sẽ càng nhiều
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    range1 = (20, 10, 50)
    range2 = (90, 255, 255)
    image_hsv = cv2.inRange(image_hsv, range1, range2)
    image_hsv = 255 - image_hsv
    # cv2.imshow("mask",cv2.resize(image_hsv,(int(image_hsv.shape[1]*0.2),int(image_hsv.shape[0]*0.2))))

    contours_hsv, hierarchy_hsv = cv2.findContours(image_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_max_hsv = max(contours_hsv, key=cv2.contourArea)
    x_hsv, y_hsv, w_hsv, h_hsv = cv2.boundingRect(contour_max_hsv)

    image = image[y_hsv:y_hsv + h_hsv, x_hsv:x_hsv + w_hsv]

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 140, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_max = max(contours, key=cv2.contourArea)

    cv2.drawContours(thresh, [contour_max], -1, 255, thickness=-1)
    # cv2.imshow("image_thresh_hhh",cv2.resize(thresh,(int(thresh.shape[1]*0.2),int(thresh.shape[0]*0.2))))

    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=3)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    x_c, y_c, w_c, h_c = cv2.boundingRect(contour_max)
    image = image[y_c:y_c + h_c, x_c:x_c + w_c]
    # cv2.imshow("image",cv2.resize(image,(int(image.shape[1]*0.2),int(image.shape[0]*0.2))))

    image_crop_thresh = thresh[y_c:y_c + h_c, x_c:x_c + w_c]
    h, w = image_crop_thresh.shape[:2]

    contours_1, hierarchy_1 = cv2.findContours(image_crop_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_max_1 = max(contours_1, key=cv2.contourArea)

    cv2.drawContours(image_crop_thresh, [contour_max_1], -1, 255, thickness=-1)

    epsilon = alpha * cv2.arcLength(contour_max_1, True)
    approx = cv2.approxPolyDP(contour_max_1, epsilon, True)

    # # h,w = image.shape[:2]
    # image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # contour_max = max(contours, key = cv2.contourArea)
    # cv2.drawContours(thresh,[contour_max], -1, 255, thickness=-1)
    # # cv2.imshow("image_thresh",cv2.resize(thresh,(int(thresh.shape[1]*0.2),int(thresh.shape[0]*0.2))))

    # epsilon = alpha*cv2.arcLength(contour_max,True)
    # approx = cv2.approxPolyDP(contour_max,epsilon,True)

    # lấy ra 4 điểm thuộc 4 góc của bảng mạch
    w_min = w / 4
    h_min = h / 4
    for i in range(len(approx)):
        if approx[i][0][0] < w_min and approx[i][0][1] < h_min:
            x_left_top, y_left_top = approx[i][0][0], approx[i][0][1]
            # image = cv2.circle(image, (approx[i][0][0],approx[i][0][1]), 30, (255,0,0), -1)
        elif approx[i][0][0] > w_min and approx[i][0][1] < h_min:
            x_right_top, y_right_top = approx[i][0][0], approx[i][0][1]
            # image = cv2.circle(image, (approx[i][0][0],approx[i][0][1]), 30, (255,0,0), -1)
        elif approx[i][0][0] < w_min and approx[i][0][1] > h_min:
            x_left_bottom, y_left_bottom = approx[i][0][0], approx[i][0][1]
            # image = cv2.circle(image, (approx[i][0][0],approx[i][0][1]), 30, (255,0,0), -1)
        elif approx[i][0][0] > w_min and approx[i][0][1] > h_min:
            x_right_bottom, y_right_bottom = approx[i][0][0], approx[i][0][1]
            # image = cv2.circle(image, (approx[i][0][0],approx[i][0][1]), 30, (255,0,0), -1)

    image_corner = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    image_crop = np.float32([[x_left_top, y_left_top], [x_right_top, y_right_top], [x_left_bottom, y_left_bottom],
                             [x_right_bottom, y_right_bottom]])

    image_output = cv2.getPerspectiveTransform(image_crop, image_corner)
    image_output = cv2.warpPerspective(image, image_output, (w, h))

    # lấy khối màu vàng và xác định tâm của khối đó
    # xác định khoảng cách của khối đến các cạnh
    # thực hiện xoay hình
    image_output_yellow = cv2.cvtColor(image_output, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([22, 93, 0])
    high_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(image_output_yellow, low_yellow, high_yellow)

    contours_yellow, hierachy_yellow = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours_yellow, key=cv2.contourArea)

    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    y, x = image_output.shape[:2]
    y_min = y / 2
    x_min = x / 2
    # cv2.drawContours(image_output, [cnt], -1, (0, 255, 0), 2)
    # cv2.circle(image_output,(cx,cy),10,(255,0,0),-1)

    # kiểm tra các khoảng cách từ tâm hình màu vàng để xoay ảnh
    if x > y:
        image_corner_1 = np.float32([[0, 0], [y, 0], [0, x], [y, x]])
        if cx < x_min and cy < y_min:
            image_crop_1 = np.float32([[x, 0], [x, y], [0, 0], [0, y]])
            image_output_1 = cv2.getPerspectiveTransform(image_crop_1, image_corner_1)
            image_output = cv2.warpPerspective(image_output, image_output_1, (y, x))
        elif cx > x_min and cy > y_min:
            image_crop_1 = np.float32([[0, y], [0, 0], [x, y], [x, 0]])
            image_output_1 = cv2.getPerspectiveTransform(image_crop_1, image_corner_1)
            image_output = cv2.warpPerspective(image_output, image_output_1, (y, x))
    elif cx > x_min and cy < y_min:
        image_corner_1 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
        image_crop_1 = np.float32([[x, y], [0, y], [x, 0], [0, 0]])
        image_output_1 = cv2.getPerspectiveTransform(image_crop_1, image_corner_1)
        image_output = cv2.warpPerspective(image_output, image_output_1, (x, y))

    # cv2.imshow("image_yellow", cv2.resize(yellow_mask,(int(yellow_mask.shape[1]*0.2),int(yellow_mask.shape[0]*0.2))))

    # Crop khung chứa phần dây
    y_1, x_1 = image_output.shape[:2]
    image_error = image_output[int(y_1 * alpha_y1):int(y_1 * alpha_y2), int(x_1 * alpha_x1):int(x_1 * alpha_x2)]
    # cv2.imshow("image_error",image_error)
    # cv2.imshow("image_1",cv2.resize(image,(int(image.shape[1]*0.2),int(image.shape[0]*0.2))))
    # cv2.imshow("image_crop",cv2.resize(image_output,(int(image_output.shape[1]*0.2),int(image_output.shape[0]*0.2))))

    return image_error, image_output, int(x_1 * alpha_x1), int(y_1 * alpha_y1)


# Hàm chỉnh sáng tối cho ảnh
# Gamma < 1 thì ảnh tăng độ sáng
# Gamma >1 thì chỉnh ảnh tối hơn
def adjust_image_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# mask red lấy dây màu đỏ
def mask_red(image):
    low_red = np.array([161, 155, 40])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(image, low_red, high_red)
    # cv2.imshow("image_red", red_mask)
    return red_mask


# mask black lấy dây màu đen
def mask_black(thresh):
    point_choose = []
    len_choose = []
    img_fillFlood = thresh.copy()
    h, w = img_fillFlood.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    while True:
        masks = np.where(img_fillFlood == 255)
        if len(masks[0]) == 0:
            break
        y, x = masks[0][0], masks[1][0]
        cv2.floodFill(img_fillFlood, mask, (x, y), 0)
        mask_1 = np.where(img_fillFlood == 255)
        len_choose.append(len(masks[0]) - len(mask_1[0]))
        point_choose.append((x, y))

    point_choose = np.array(point_choose)
    len_choose = np.array(len_choose)
    mask_1 = np.where(len_choose < 300)
    # print(mask_1)
    tmp1 = thresh.copy()
    mask1 = np.zeros((h + 2, w + 2), np.uint8)

    for idx in mask_1[0]:
        x, y = point_choose[idx]
        cv2.floodFill(thresh, mask1, (x, y), 0)

    return thresh


# hàm tìm điểm đầu và cuối dây 1
def findPoint1(mask, width):
    x_left, y_left, x_right, y_right = width, width, 0, 0
    corners = cv2.goodFeaturesToTrack(mask, 20, 0.1, 10)
    for i in corners:
        x, y = i.ravel()
        if x < x_left:
            x_left, y_left = x, y
        elif x > x_right:
            x_right, y_right = x, y
    return x_left, y_left, x_right, y_right


# hàm tìm điểm đầu và điểm cuối dây 2
def findPoint2(mask, width, height, max_point=30):
    x_left, y_left = width, height
    y_left_min = y_left / 4
    corners = cv2.goodFeaturesToTrack(mask, max_point, 0.1, 10)
    for i in corners:
        x, y = i.ravel()
        # cv2.circle(image, (x,y), 5, (0,0,255), -1)
        if y < y_left and y > y_left_min:
            x_left, y_left = x, y
    return x_left, y_left


# Hàm lấy dây màu trắng hoặc đen
def mask_color(thresh):
    point_choose = []
    len_choose = []
    img_fillFlood = thresh.copy()
    h, w = img_fillFlood.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    while True:
        masks = np.where(img_fillFlood == 255)
        if len(masks[0]) == 0:
            break
        y, x = masks[0][0], masks[1][0]
        cv2.floodFill(img_fillFlood, mask, (x, y), 0)
        mask_1 = np.where(img_fillFlood == 255)
        len_choose.append(len(masks[0]) - len(mask_1[0]))
        point_choose.append((x, y))

    point_choose = np.array(point_choose)
    len_choose = np.array(len_choose)
    # lấy giá trị nhỏ hơn ía trị max trừ đi
    len_choose_max = 0

    # lấy giá trị dài nhất
    for i in range(len(len_choose)):
        if len_choose_max < len_choose[i]:
            len_choose_max = len_choose[i]

    mask_1 = np.where(len_choose < len_choose_max)
    tmp1 = thresh.copy()
    mask1 = np.zeros((h + 2, w + 2), np.uint8)
    for idx in mask_1[0]:
        x, y = point_choose[idx]
        cv2.floodFill(thresh, mask1, (x, y), 0)

    # cv2.imshow("thresh",thresh)
    return thresh


# Hàm crop phần ảnh mối hàn
def image_crop3(image):
    # crop phần chứa bảng mạch
    # tìm countour bảng mạch sau đó crop
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_max = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour_max)
    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    image_crop = image[y:y + h, x:x + w]

    # Crop khung chứa phần dây
    # tính tỉ lệ khung dây cần tìm lỗi
    y_crop, x_crop = image_crop.shape[:2]
    image_e = image_crop[int(y_crop * 0.6023):int(y_crop * 0.7124), int(x_crop * 0.4753):int(x_crop * 0.5565)]

    return image_crop, image_e, int(x_crop * 0.4753), int(y_crop * 0.6023)


# Hàm crop phần ảnh ốc vít
def image_crop4(image):
    # crop phần chứa bảng mạch
    # tìm countour bảng mạch sau đó crop
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_max = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour_max)
    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    image_crop = image[y:y + h, x:x + w]

    # Crop khung chứa phần dây
    # tính tỉ lệ khung dây cần tìm lỗi
    y_crop, x_crop = image_crop.shape[:2]
    image_e = image_crop[int(y_crop * 0.49):int(y_crop * 0.6124), int(x_crop * 0.01):int(x_crop * 0.9)]

    return image_crop, image_e, int(x_crop * 0.01), int(y_crop * 0.49)


# Ham Resize anh
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def non_overlap(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def findboxes(image):
    extract = True
    height_image_e,width_image_e,_ = image_e.shape
    image_e = cv2.resize(image_e,(152,258))
    # chỉnh độ sáng của ảnh trước khi tìm dây đen
    #  tim dây đen trong khung hình
    image_const_1 = adjust_image_gamma(image_e,1.1)
    image_const_1 = cv2.cvtColor(image_const_1,cv2.COLOR_BGR2GRAY)
    ret_1, thresh_1 = cv2.threshold(image_const_1, 50, 255, cv2.THRESH_BINARY_INV)
    black_mask = mask_black(thresh_1)
    # chỉnh dộ sáng của ảnh trước khi tìm box 
    #  tìm box vẽ bên ngoài chữ white và dấu +
    # hoặc box vẽ bên ngoài chữ black và dấu - 
    image_const = adjust_image_gamma(image_e,1.0)
    image_const = cv2.cvtColor(image_const,cv2.COLOR_BGR2GRAY)
    ret,thresh_2 = cv2.threshold(image_const,120,255,cv2.THRESH_BINARY_INV)

    # xác định box 
    #  tìm countour rồi xác định 
    contours, hierarchy = cv2.findContours(thresh_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_right_bottom,y_right_bottom = 0,0 
    x_left_bottom,y_left_bottom = 0,0

    for contour in contours:
        # cv2.drawContours(thresh_2,[contour], -1, 255, thickness=-1)
        if 450<cv2.contourArea(contour) <2000:
            x_1,y_1,w,h = cv2.boundingRect(contour)
            if x_1+w < image_e.shape[1]*0.45: 
                x_right_bottom,y_right_bottom = x_1+w,y_1+h
                image_e = cv2.rectangle(image_e,(x_1,y_1),(x_1+w,y_1+h),(0,255,0),2)
                cv2.imshow("image_1",image_e)
            elif x_1+w > image_e.shape[0]*0.45:
                x_left_bottom,y_left_bottom = x_1,y_1
                image_e = cv2.rectangle(image_e,(x_1,y_1),(x_1+w,y_1+h),(0,255,0),2)
                cv2.imshow("image_2",image_e)
  
    # tìm point đầu và cuối của dây đen
    # lấy ra 4 tọa độ point rồi vẽ vào hình
    x_left_black,y_left_black = findPoint2(black_mask,image_e.shape[1],image_e.shape[0])
    image_e = cv2.circle(image_e, (x_left_black,y_left_black), 6, (255,0,0), -1)
    image_e = cv2.circle(image_e, (x_right_black,y_right_black), 6, (255,0,0), -1)
    # cv2.imshow("image",image_e)
    # điều kiện để  xác định nối sai 
    # so sánh tọa độ x với điểm box bên trái của chữ black 
    # hoặc điểm box bên phải của chữ white
    if x_left_black < x_right_bottom and x_right_bottom != 0:
        print("connect error")
        extract = False
    elif x_left_black < x_left_bottom and x_left_bottom != 0:
        print("connect error")
        extract = False
    else:
        print("connect true")

    return extract
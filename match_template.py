import cv2
import numpy as np
import os


def template_match_drawline(_img_path, _template_path, is_only_max=True, method="SQDIFF", threshold=0.9):

    img = cv2.imread(_img_path)
    template = cv2.imread(_template_path)

    height, width, channel = template.shape

    if method == "SQDIFF":
        if is_only_max:
            res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            x = min_loc[0]
            y = min_loc[1]
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.imshow("SQDIFF", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            res = cv2.matchTemplate(img, template, cv2.cv2.TM_SQDIFF_NORMED)
            threshold = 1-threshold
            loc = np.where(res <= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img, pt, (pt[0] + width, pt[1] + height), (0, 0, 255), 2)
            cv2.imshow("SQDIFF", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif method == "CCORR":
        if is_only_max:
            res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            x = max_loc[0]
            y = max_loc[1]
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.imshow("CCORR", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
            print(res.shape)
            print(res[0])
            print(res[0].shape)
            loc = np.where(res >= threshold)
            print(*loc[::-1])

            for pt in zip(*loc[::-1]):

                cv2.rectangle(img, pt, (pt[0] + width, pt[1] + height), (0, 0, 255), 2)
            cv2.imshow("CCORR", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif method == "CCOEFF":
        if is_only_max:
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            x = max_loc[0]
            y = max_loc[1]
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.imshow("CCOEFF", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img, pt, (pt[0] + width, pt[1] + height), (0, 0, 255), 2)
            cv2.imshow("CCOEFF", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def template_match_crop(_img_path, _template_path, res_dir, is_only_max=True, method="SQDIFF", threshold=0.9):
    _, file= os.path.split(_img_path)
    img = cv2.imread(_img_path)
    template = cv2.imread(_template_path)
    height, width, channel = template.shape

    if method == "SQDIFF":
        if is_only_max:
            res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            x = min_loc[0]
            y = min_loc[1]
            crop_img = img[y:y+height, x:x+width]
            cv2.imwrite(os.path.join(res_dir, "SQDIFF_"+file), crop_img)

        else:
            res = cv2.matchTemplate(img, template, cv2.cv2.TM_SQDIFF_NORMED)
            threshold = 1-threshold
            loc = np.where(res <= threshold)
            count = 1
            for pt in zip(*loc[::-1]):
                crop_img = img[pt[1]:pt[1] + height, pt[0]:pt[0] + width]
                cv2.imwrite(os.path.join(res_dir, "SQDIFF_"+str(count)+"_" + file), crop_img)
                count += 1

    elif method == "CCORR":
        if is_only_max:
            res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            x = max_loc[0]
            y = max_loc[1]
            crop_img = img[y:y + height, x:x + width]
            cv2.imwrite(os.path.join(res_dir, "CCORR_" + file), crop_img)
        else:
            res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
            loc = np.where(res >= threshold)
            count = 1
            for pt in zip(*loc[::-1]):
                crop_img = img[pt[1]:pt[1] + height, pt[0]:pt[0] + width]
                cv2.imwrite(os.path.join(res_dir, "CCORR_"+str(count)+"_"+ file), crop_img)
                count += 1
    elif method == "CCOEFF":
        if is_only_max:
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            x = max_loc[0]
            y = max_loc[1]
            crop_img = img[y:y + height, x:x + width]
            cv2.imwrite(os.path.join(res_dir, "CCOEFF_" + file), crop_img)
        else:
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            count = 1
            for pt in zip(*loc[::-1]):
                crop_img = img[pt[1]:pt[1] + height, pt[0]:pt[0] + width]
                cv2.imwrite(os.path.join(res_dir, "CCOEFF_"+str(count)+"_" + file), crop_img)
                count += 1


if __name__ == '__main__':
    src_path = r".\img_match_template\img.png"
    template_path = r".\img_match_template\template.jpg"
    res_dir = r".\img_match_template\res"

    template_match_drawline(_img_path=src_path, _template_path=template_path, is_only_max=True, method="SQDIFF")
    template_match_drawline(_img_path=src_path, _template_path=template_path, is_only_max=False, method="SQDIFF")

    template_match_drawline(_img_path=src_path, _template_path=template_path, is_only_max=True, method="CCOEFF")
    template_match_drawline(_img_path=src_path, _template_path=template_path, is_only_max=False, method="CCOEFF")

    template_match_drawline(_img_path=src_path, _template_path=template_path, is_only_max=True, method="CCORR")
    template_match_drawline(_img_path=src_path, _template_path=template_path, is_only_max=False, method="CCORR"
                          , threshold=0.95)

    template_match_crop(_img_path=src_path, _template_path=template_path, is_only_max=True, method="SQDIFF",
                        res_dir=res_dir)
    template_match_crop(_img_path=src_path, _template_path=template_path, is_only_max=False, method="SQDIFF",
                        res_dir=res_dir)

    template_match_crop(_img_path=src_path, _template_path=template_path, is_only_max=True, method="CCOEFF",
                        res_dir=res_dir)
    template_match_crop(_img_path=src_path, _template_path=template_path, is_only_max=False, method="CCOEFF",
                        res_dir=res_dir)

    template_match_crop(_img_path=src_path, _template_path=template_path, is_only_max=True, method="CCORR",
                        res_dir=res_dir)
    template_match_crop(_img_path=src_path, _template_path=template_path, is_only_max=False, method="CCORR"
                        , threshold=0.95, res_dir=res_dir)


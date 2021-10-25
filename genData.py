from Split1 import splitImg1
from split0907 import splitImg



################# 切割图片超参数 ###################
PIC_PATH = 'G:/liudongbo/projects/8-25/210902/'                # 原图路径(文件夹)
PIC_NUM = 2                         # 原图数量
SAVE_PATH = 'G:/liudongbo/dataset/small_pic_test/'             # 切出来的小图保存路径
SUFFIX = 'tif'                      # 图片后缀,ex:'tif', 'jpg', 'png','tiff'
PX = 7                              # 切出来的小图的边宽,12卷两次池化一次变6


THRESHVAL = 200
SETEXCEPTAREAVAL = []
SETM = -1
SUMTHRESHVAL = 100

TXT_NAME = '0902data.txt'

TEST_SIZE = 0.1                    # 测试集占总数据的比例

IMG_TYPE = "both"   #  ["delay","plr","both"]

# n_split = splitImg1(pic_path=PIC_PATH,pic_num= PIC_NUM,suffix= SUFFIX,ThreshVal=THRESHVAL,SetExceptAreaVal=SETEXCEPTAREAVAL,
#                     SetM = SETM,px=PX,save_path=SAVE_PATH,SumThreshVal=SUMTHRESHVAL,txtName = TXT_NAME,test_size = TEST_SIZE)

n_split = splitImg(pic_path=PIC_PATH,pic_num= PIC_NUM,suffix= SUFFIX,ThreshVal=THRESHVAL,SetExceptAreaVal=SETEXCEPTAREAVAL,
                    SetM = SETM,px=PX,save_path=SAVE_PATH,SumThreshVal=SUMTHRESHVAL,txtName = TXT_NAME,test_size = TEST_SIZE,imgType = IMG_TYPE)



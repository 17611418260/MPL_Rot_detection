# from ultralytics.models.yolo import YOLO
# from trainer import yolov8_MPL_Trainer
# from ultralytics import YOLO
import sys
# sys.path.append("..") 
from ultralytics import YOLO
import pdb
# from yolov8
# from .. import ultralytics
# from ..ultralytics import YOLO
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='/data/jingshengzeng/Rot_obj_det/jch/cocotest.yaml', help='dataset.yaml path')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
parser.add_argument('--hyp', type=str, default='/data/jingshengzeng/Rot_obj_det/jch/hyp.yaml', help='size of each image batch')
parser.add_argument('--model', type=str, default='/data/jingshengzeng/Rot_obj_det/jch/yolov8x.pt', help='pretrained weights or model.config path')
# /data/chenghanjia/yolov8/jch/yolov8x.pt
# /data/chenghanjia/yolov8/yolo/pretrain_nodot_conf0.7/weights/yolov8x.pt
parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
parser.add_argument('--img-size', type=int, default=640, help='size of each image dimension')
parser.add_argument('--device', type=str, default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--workers', type=int, default=8, help='number of worker threads for data loading (per RANK if DDP)')
parser.add_argument('--project', type=str, default='yolo', help='project name')
parser.add_argument('--name', type=str, default='yolov8_pl', help='exp name') # pretrain\\yolov8_mpl_conf0.8_finlabel_4set_epoch900
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--teacher-warmup',type=int, default=0, help='教师网络预热轮数')
parser.add_argument('--teacher-model', type=str, default='/data/jingshengzeng/Rot_obj_det/yolo/teacher_model_pretrain/TGRS_yolov8_org/weights/best.pt', help='教师网络预热权重')
parser.add_argument('--pserdo-label-conf',type=float, default=0.8, help='生成伪标签时对应的NMS置信度门限')
parser.add_argument('--lambda-u', default=8, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-epoch', default=50, type=float, help='warmup epoch of lambda-u')
parser.add_argument('--fin-student-epoch', default=40, type=float, help='final student train with label set')
parser.add_argument('--close-mosaic', default=120, type=float, help='the epoch when close mosaic argument')

args = parser.parse_args()

assert args.data, 'argument --data path is required'
assert args.model, 'argument --model path is required'

if __name__ == '__main__':
    # Initialize
    model = YOLO(args.model)
    # model = YOLO()
    
    hyperparams = yaml.safe_load(open(args.hyp))
    hyperparams['epochs'] = args.epochs
    hyperparams['batch'] = args.batch_size
    hyperparams['imgsz'] = args.img_size
    hyperparams['device'] = args.device
    hyperparams['workers'] = args.workers
    hyperparams['project'] = args.project
    hyperparams['name'] = args.name
    hyperparams['resume'] = args.resume
    hyperparams['teacher_warmup'] = args.teacher_warmup
    hyperparams['teacher_model'] = args.teacher_model
    hyperparams['pserdo_label_conf'] = args.pserdo_label_conf
    hyperparams['lambda_u'] = args.lambda_u
    hyperparams['uda_epoch'] = args.uda_epoch
    hyperparams['fin_student_epoch'] = args.fin_student_epoch
    hyperparams['close_mosaic'] = args.close_mosaic
    model.train(data= args.data, **hyperparams)
    # yolov8_MPL_Trainer(data= args.data, **hyperparams)


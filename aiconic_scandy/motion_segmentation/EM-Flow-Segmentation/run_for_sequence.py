from Models.CoherenceNets.MethodeB import MethodeB
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


def read_flo(path):
    with open(path, "rb") as f:
        (magic,) = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            raise IOError("Magic number incorrect. Invalid .flo file")
        else:
            w, h = np.fromfile(f, np.int32, count=2)
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (h, w, 2))
            # flow = np.swapaxes(flow, 0, 1)
    return flow


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("Using CUDA")
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("NOT using CUDA")
        torch.set_default_tensor_type(torch.FloatTensor)
    ckpt = '/media/vito/TOSHIBA EXT/em-driven-segmentation-data-main-2-Masks ( 1tdcjfqp )-checkpoint/2-Masks ( 1tdcjfqp )/checkpoint/epoch=31-epoch=epoch_val_loss=0.53934.ckpt'# Path of the model
    model = MethodeB.load_from_checkpoint(ckpt, strict=False).eval()

    for i in range(100):
        flow_path = os.path.join(
            "/media/vito/TOSHIBA EXT/scanpath_data/1cIrRGn5mmQ", "flows", f"{(i):04d}.flo"
        )
        orig_flow = read_flo(flow_path)
        orig_size = (640, 480)
        flow = cv2.resize(orig_flow, (model.hparams['img_size'][1],model.hparams['img_size'][0]))
        flow =torch.swapaxes(torch.from_numpy(flow).unsqueeze(0), 0, 3).squeeze().unsqueeze(0)
        print("next picture")
        while torch.sum(torch.abs(flow) > 3.0) >0:
            with torch.no_grad() :
                r = model.prediction({'Flow' :flow.type(torch.get_default_dtype()).to(torch.cuda.current_device())})
            result_mask = r['Pred'].argmax(1)[0]
            plt.imshow((result_mask.cpu().numpy()))
            plt.show()
            flow[:, :, result_mask>0] = 0
            print("one mask done")
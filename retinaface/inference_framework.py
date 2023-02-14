import torch
import numpy as np

torch.set_grad_enabled(False)

# My libs
import retinaface.models.retinaface as rf_model
import retinaface.detect as rf_detect
import retinaface.data.config as rf_config
import retinaface.layers.functions.prior_box as rf_priors
import retinaface.utils.box_utils as rf_ubox
import retinaface.utils.nms.py_cpu_nms as rf_nms


# Default configs
cfg_postreat_dft = {'resize': 1.,
                    'score_thr': 0.75,
                    'top_k': 5000,
                    'nms_thr': 0.4,
                    'keep_top_k': 50}


class RetinaFaceDetector:

    def __init__(self,
                 model='mobile0.25',
                 device='cuda',
                 extra_features=['landmarks'],
                 cfg_postreat=cfg_postreat_dft):

        # Set model configuration
        cfg = None
        trained_model = None
        if model == "mobile0.25":
            cfg = rf_config.cfg_mnet
            trained_model = "https://drive.google.com/uc?export=download&confirm=yes&id=1nxhtpdVLbmheUTwyIb733MrL53X4SQgQ"
            url_model_name = "retinaface_mobile025.pth"
        elif model == "resnet50":
            cfg = rf_config.cfg_re50
            trained_model = "https://drive.google.com/uc?export=download&confirm=yes&id=1a9SqFRkeTuJUwqerElCWJFrotZuDGVtT"
            url_model_name = "retinaface_resnet50.pth"
        else:
            raise ValueError('Model configuration not found')

        # Load net and model
        cpu_flag = 'cpu' in device
        net = rf_model.RetinaFace(cfg=cfg, phase='test')
        net = rf_detect.load_model(net, trained_model, cpu_flag, url_file_name=url_model_name)
        net.eval()
        print('RetinaFace loaded!')

        # Define detector variables
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.cfg = cfg
        self.features = ['bbox'] + extra_features
        self.scale = {}
        self.prior_data = None

        # Postreatment configuration
        self.cfg['postreat'] = cfg_postreat

    def set_input_shape(self, im_height, im_width):

        # Scales
        scale_bbox = torch.Tensor([im_width, im_height, im_width, im_height])
        self.scale['bbox'] = scale_bbox.to(self.device)

        if 'landmarks' in self.features:
            scale_lnd = torch.Tensor([im_width, im_height, im_width, im_height,
                                      im_width, im_height, im_width, im_height,
                                      im_width, im_height])
            self.scale['landmarks'] = scale_lnd.to(self.device)

        # Load priors
        priorbox = rf_priors.PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        self.prior_data = priors.data

    def inference(self, image):
        img = self._pretreatment(image)
        loc, conf, lnd = self._net_forward(img)
        features = self._postreatment(loc, conf, lnd)
        return features

    def _pretreatment(self, img_raw):
        img = np.float32(img_raw)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        return img

    def _net_forward(self, img):
        loc, conf, landms = self.net(img)
        return loc, conf, landms

    def _postreatment(self, loc, conf, landms):

        cfg_post = self.cfg['postreat']
        boxes = rf_ubox.decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
        boxes = boxes * self.scale['bbox'] / cfg_post['resize']
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = rf_ubox.decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
        landms = landms * self.scale['landmarks'] / cfg_post['resize']
        landms = landms.cpu().numpy()

        # Ignore low scores
        inds = np.where(scores > cfg_post['score_thr'])[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # Keep top-K before NMS
        order = scores.argsort()[::-1][:cfg_post['top_k']]
        boxes = boxes[order]
        scores = scores[order]

        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = rf_nms.py_cpu_nms(dets, cfg_post['nms_thr'])
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:cfg_post['keep_top_k'], :]

        features = {'bbox': dets}
        if 'landmarks' in self.features:
            landms = landms[inds]
            landms = landms[order]
            landms = landms[keep]
            landms = landms[:cfg_post['keep_top_k'], :]
            landms = np.array(landms)
            landms = np.expand_dims(landms, axis=-1)
            landms = landms.reshape((-1, 5, 2))
            features['landmarks'] = landms

        return features

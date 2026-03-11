from .baseline import BaselinePredictor
from isegm.inference.transforms import ZoomIn

try:
    from .cdnet import DiffisionPredictor
    from .focalclick import FocalPredictor
    from .brs import InputBRSPredictor, FeatureBRSPredictor, HRNetFeatureBRSPredictor
    from .brs_functors import InputOptimizer, ScaleBiasOptimizer
    from isegm.model.is_hrnet_model import HRNetModel
    _full_import = True
except Exception:
    _full_import = False
    HRNetModel = None


def get_predictor(net, brs_mode, device,
                  prob_thresh=0.49,
                  infer_size=256,
                  focus_crop_r=1.4,
                  with_flip=False,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  brs_opt_func_params=None,
                  lbfgs_params=None):
    lbfgs_params_ = {
        'm': 20,
        'factr': 0,
        'pgtol': 1e-8,
        'maxfun': 20,
    }
    predictor_params_ = {
        'optimize_after_n_clicks': 1
    }

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if brs_mode in ('NoBRS', 'Baseline'):
        if predictor_params is not None:
            predictor_params_.update(predictor_params)
        predictor = BaselinePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip, infer_size=infer_size, **predictor_params_)
    else:
        raise NotImplementedError(f"brs_mode '{brs_mode}' requires full ClickSEG install (mmcv etc.)")

    return predictor

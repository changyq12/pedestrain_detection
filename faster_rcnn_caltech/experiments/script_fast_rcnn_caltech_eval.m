function script_fast_rcnn_caltech_eval()

clc;
clear mex;
clear is_valid_handle; 
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

%conf.exp_name='faster_rcnn_stage1'
% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_Faster_RCNN_VOC2007;

dataset                     = [];

dataset                     = Dataset.caltech_trainval(dataset, 'train');
dataset                     = Dataset.caltech_test(dataset, 'test');

conf_fast_rcnn              = fast_rcnn_config('image_means', model.mean_image);


cache_name='caltech';
method_name='FRCN-stage2';
conf_fast_rcnn.exp_name='faster-rcnn-caltech';

do_nms=1;        
model.nms.per_nms_topN=-1;
model.nms.nms_overlap_thres=0.5; 
model.nms.after_nms_topN=-1;

Faster_RCNN_Train.do_faster_rcnn_test_eval(conf_fast_rcnn, do_nms,model, dataset.imdb_test, dataset.roidb_test, cache_name, method_name);

end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file);
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  2.^[3:5]);
end

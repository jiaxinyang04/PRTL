clear
close all

rng('default') % For reproducibility

addpath(genpath('utils/'))
data_path = 'data/';
Data_list = 'MSRCv1_5v.mat';
load(fullfile(data_path, Data_list));

gt=Y;
K = length(X); % the number of views
cls_num = length(unique(gt)); % the number of clusters
for k=1:K
    X{k}=X{k}';
    [X{k}]=NormalizeData(X{k}); 
end

opts = [];
opts.maxIter = 120;
opts.mul_rate = 1.1;    
opts.nb_num   = 8;      
opts.yita     = 0.01;
                
[Out,obj] = PRTL(X, cls_num, gt, opts);

fprintf('\n%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\n',...\
        ' ACC','NMI','PUR','AR','Recall','Pre','FScore','mulrate','nbnum', 'yita');
fprintf('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
         Out.ACC,Out.NMI,Out.PUR,Out.AR, Out.recall,Out.precision,Out.fscore,opts.mul_rate,opts.nb_num,opts.yita);                
                   
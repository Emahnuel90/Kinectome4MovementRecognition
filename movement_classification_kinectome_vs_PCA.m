function T = movement_classification_kinectome_vs_PCA(data, labels)

% DATA_ANALYSIS_TO_PUBLISH performs dimensionality reduction and classification
% analysis on movement data using network theory and PCA, returning a summary
% table of classification accuracies.
%
% INPUTS:
%   data   - A cell matrix of size n x m, where n is the number of subjects
%            and m is the number of movement classes. Each cell contains a
%            numeric matrix representing a movement trial, with rows as 
%            time frames and columns as body parts (possibly including multiple 
%            columns per body part if movement is tracked across multiple planes).
%
%   labels - A 1 x m cell array of strings containing the names of the movement
%            classes corresponding to the columns of "data".
%
% OUTPUT:
%   T      - A table summarizing the classification results, including the mean,
%            median, and standard deviation of classification accuracy for each
%            approach used (network-based and PCA-based).
%
% DESCRIPTION:
% The function performs a two-step analysis:
%   1. **Dimensionality Reduction**:
%      - *Network Theory*: Each trial matrix is converted into a "kinectome",
%        a pairwise correlation matrix representing inter-body-part coordination.
%      - *PCA*: Principal Component Analysis is applied to the original matrices,
%        and components are retained until 95% of the total variance is explained.
%
%   2. **Classification**:
%      - An SVM (Support Vector Machine) classifier with 10-fold cross-validation
%        is used to classify movement types based on the features extracted via
%        both methods.
%
% The function outputs confusion matrices and computes summary statistics of the
% classification accuracy for both approaches, returning a table formatted for
% publication or reporting.

% -------------------------------------------------------------------------
% Authors: Emahnuel Troisi Lopez, Marie-Constance Corsi
% Date:    20/06/2025
% License: Creative Commons Attribution 4.0 International (CC BY 4.0)
%          https://creativecommons.org/licenses/by/4.0/
%
% You are free to share and adapt the code, even for commercial purposes,
% as long as proper credit is given.
% -------------------------------------------------------------------------
% LINK AL DATASET
%
% LINK AD ARXIV

%% Further optional settings
%choosing the number of k-folds for cross-validation
nkfold = 10;

% Choosing variance explained by PCA features to determine how many features should be considered
exp_var = 95;

derivatives = 0; % if 0 data doesn't change. If 1 or more to calculate the corresponding derivative of the time series
holes = 0;  % if 0 data doesn't change. If 1 random NaN are applied to the data.
%NaN length is equal to 1/10 of the total data length and, for each recording occurs, is applied at a random frame for 3 random joints.

%% DATA MANAGEMENT
for zz1 = 1:size(data,1)
    for zz2 = 1:size(data,2)
        if derivatives > 0
            data{zz1,zz2} = diff(data{zz1,zz2}, derivatives);
        end
        if holes == 1
            holength = round(size(data{1,1},1)/10); %Length of the holes
            joints = randperm(12,3); %find 3 random joints
            startingholes = randperm(68-(holength-1),3); %3 random starting points for the holes
            temp = data{zz1,zz2};
            for zz3 = 1:length(joints)
                temp(startingholes(zz3):startingholes(zz3)+holength-1,joints(zz3)*3-2:joints(zz3)*3) = NaN;
            end
            data{zz1,zz2} = temp;
        end
    end
end
njoints = size(data{1},2);
matjoints = zeros(njoints);
mask_ut = logical(triu(ones(size(matjoints)),1));
MOVS_kin = zeros(size(data,1), size(data,2), nnz(mask_ut));
for subj = 1:size(data,1)
    for mov = 1:size(data,2)
    temp = corr(data{subj,mov}(:,:,1),'Rows','pairwise');
    MOVS_kin(subj,mov,:) = temp(mask_ut)';
    end
end

nframes = size(data{1},1);
MOVS_angles = zeros(size(data,1), size(data,2), nframes*njoints);
for subj = 1:size(data,1)
    for mov = 1:size(data,2)
        temp = data{subj,mov}(:,:,1);
        temp = temp(:);
        MOVS_angles(subj,mov,:) = temp;
    end
end

%% START
% First, MOVS_angles and MOVS_kin are reshaped to have movements and participants on the rows
%
% The new variable (movs) will have, on the rows:
% 1) movement 1 performed by participant 1
% 2) movement 1 performed by participant 2
% 3) ...and so on
temp_movs = reshape(MOVS_angles,size(MOVS_angles,1)*size(MOVS_angles,2),size(MOVS_angles,3));
temp_movkin = reshape(MOVS_kin,size(MOVS_kin,1)*size(MOVS_kin,2),size(MOVS_kin,3));

Y = repelem(labels,size(MOVS_angles,1));
num_labels = size(Y,2);
s = RandStream('dsfmt19937');
newLabelsOrder = randperm(s,num_labels);

rng(1); % For reproducibility

movs = temp_movs(newLabelsOrder,:);
movkin = temp_movkin(newLabelsOrder,:);
Y_shuffle = Y(newLabelsOrder);


cvp = cvpartition(Y_shuffle, 'KFold', nkfold);
PCApred = cell(1, size(movs,1));
KINpred = cell(1, size(movs,1));
for zkf = 1:nkfold
    disp(zkf)
    movs_train = movs(cvp.training(zkf),:);
    movkin_train = movkin(cvp.training(zkf),:);
    Y_train = Y_shuffle(cvp.training(zkf));
    
    movs_test = movs(cvp.test(zkf),:);
    movkin_test = movkin(cvp.test(zkf),:);
    
    [trainedModel_Scale_PCA_SVM2, ~, ~] = trainClassifier_Scale_PCA_SVM_crossval(movs_train, Y_train',exp_var);
    [yfit,~] = trainedModel_Scale_PCA_SVM2.predictFcn(movs_test);
    PCApred(cvp.test(zkf)) = yfit;
    
    template = templateSVM(...
        'KernelFunction', 'linear', ...
        'PolynomialOrder', [], ...
        'KernelScale', 1, ...
        'BoxConstraint', 1, ...
        'Standardize', true);
    
    classificationSVM = fitcecoc(...
        movkin_train, ...
        Y_train, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', labels);
    
    KINpred(cvp.test(zkf)) = predict(classificationSVM, movkin_test);
end

%% OUTPUT GENERATION
ACCU_PCA_sep = zeros(size(labels));
for zzmovna = 1:length(labels)
    masks = strcmp(Y_shuffle,labels{zzmovna});
    ACCU_PCA_sep(zzmovna) = sum(strcmp(Y_shuffle(masks),PCApred(masks))) / size(MOVS_angles,1);
end

ACCU_KIN_sep = zeros(size(labels));
for zzmovna = 1:length(labels)
    masks = strcmp(Y_shuffle,labels{zzmovna});
    ACCU_KIN_sep(zzmovna) = sum(strcmp(Y_shuffle(masks),KINpred(masks))) / size(MOVS_angles,1);
end

allres = {ACCU_PCA_sep, ACCU_KIN_sep};
rescol = {[0 0 1],[1 .5 0]};
figure, hold on
for zz1 = 1:size(allres,2)
    swarmchart(zz1*ones(1,length(ACCU_KIN_sep)),allres{zz1},[],rescol{zz1},'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);
end
xticks(1:2)
xticklabels({'PCA','KIN'})
xlim([0 3])
ylabel('Accuracy')
ylim([0 1.02])
title('Accuracy for 30 movements classification')
title('Acceleration')
axis square

%Visual output
figure
ConfMat = confusionchart(Y_shuffle, PCApred);
ConfMat.RowSummary = 'row-normalized';
title([' PCA - SVM - shuffle | variance: ' num2str(exp_var)])

figure
ConfMat = confusionchart(Y_shuffle, KINpred);
ConfMat.RowSummary = 'row-normalized';
title(' KIN - SVM - shuffle')

mean_accu = mean(ACCU_PCA_sep);
median_accu = median(ACCU_PCA_sep);
std_accu = std(ACCU_PCA_sep);

mean_kin = mean(ACCU_KIN_sep);
median_kin = median(ACCU_KIN_sep);
std_kin = std(ACCU_KIN_sep);

% Output in table
T = table( ...
    [mean_accu; median_accu; std_accu], ...
    [mean_kin; median_kin; std_kin], ...
    'VariableNames', {'ACCU_PCA_sep', 'KIN_PCA_sep'}, ...
    'RowNames', {'Mean', 'Median', 'StdDev'});
end
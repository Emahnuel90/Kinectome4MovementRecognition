function yPred = umap_svm_cv_classifier(Xtr, Xte, ytr, labels, params)


% UMAP + SVM con 10-fold CV, senza data leakage
% Dati attesi:
%   X : [N x D] double (N=5040, D=2448)
%   y : [N x 1] etichette (numeric/string/categorical) con 30 classi
%   params must be a structure including (values are an example):
% kFolds      = 10;
% umapDims    = 10;     % prova 10-50 in base al dataset
% umapNN      = 15;
% umapMinDist = 0.3;
% svmKernel   = 'rbf';
% svmBoxC     = 2;
% svmKScale   = 'auto';
% Requisito: run_umap.m nel path (UMAP per MATLAB - File Exchange).

rng(1);

% Carica/definisci X e y qui, poi assicura categorical
% load('dataset.mat','X','y');
% cats = categories(y);

% Parametri
umapDims    = params.umapDims;
umapNN      = params.umapNN;
umapMinDist = params.umapMinDist;
svmKernel   = params.svmKernel;
svmBoxC     = params.svmBoxC;
svmKScale   = params.svmKScale;


    % Z-score SOLO sul training
    mu  = mean(Xtr,1);
    sig = std(Xtr,[],1);  sig(sig==0) = 1;
    XtrZ = (Xtr - mu) ./ sig;
    XteZ = (Xte - mu) ./ sig;

    % UMAP: fit su training, template per trasformare il test
    templateFile = [tempname, '.umap.mat'];

    [Ytr, ~] = run_umap( ...
        XtrZ, ...
        'n_components', umapDims, ...
        'n_neighbors',  umapNN, ...
        'min_dist',     umapMinDist, ...
        'metric',       'euclidean', ...
        'verbose',      'none', ...
        'save_template_file', templateFile ...
    );

    Yte = run_umap( ...
        XteZ, ...
        'template_file', templateFile, ...
        'verbose', 'none' ...
    );

    % SVM multiclasse (ECOC) sui dati ridotti
    t = templateSVM('KernelFunction', svmKernel, ...
                    'Standardize', true, ...
                    'BoxConstraint', svmBoxC, ...
                    'KernelScale', svmKScale);

    mdl = fitcecoc(Ytr, ytr', 'Learners', t, ...
                   'Coding','onevsall', ...
                   'ClassNames', labels);

    [yPred, ~] = predict(mdl, Yte);


    if exist(templateFile,'file'); delete(templateFile); end
%% Cover Type dataset
%%%%%%%%%%%%%%%%%%% Preprocessing Cover Type dataset %%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

fid=fopen('covtype.txt');

data = textscan(fid,...
    '%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u',...
    'Delimiter',',');
fclose(fid);

Featuresize = zeros(40,1);
for i=1:length(Featuresize)
    Featuresize(i) = sum(data{1,i+14});
end
[~,maxInd] = max(Featuresize);

covtype_dataset = [];
for i=1:14
    covtype_dataset = horzcat(covtype_dataset, data{1,i});
end

%covtype_dataset = covtype_dataset(1:20000,:);

class = zeros(size(covtype_dataset,1),1);
largestFeature = data{1,maxInd+14};

for i=1:size(covtype_dataset,1)
    if(largestFeature(i) == 1)
        class(i) = 1;
    else
        class(i) = -1;
    end
end

covtype_datasetX = covtype_dataset;
covtype_datasetY = class;
covtype_dataset = horzcat(covtype_dataset, class);
%%%%%%%%%%%% Finish preprocessing all the data for cover type %%%%%%%%%%%%%

%%%%%%% Start seperating training data and testing data for covtype %%%%%%%

[trainInd, ~, testInd] = dividerand(size(covtype_datasetY,1),0.00430284,0,1-0.00430284);
train_X = double(covtype_datasetX(trainInd,:));
train_Y = double(covtype_datasetY(trainInd,:));
test_X = double(covtype_datasetX(testInd,:));
test_Y = double(covtype_datasetY(testInd,:));

%%%%%%% Finish seperating training data and testing data for covtype %%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Normalize Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:14
    train_X(:,i) = double((train_X(:,i)-min(train_X(:,i))))/double(max(train_X(:,i))-min(train_X(:,i)));
    test_X(:,i) = double((test_X(:,i)-min(test_X(:,i))))/double(max(test_X(:,i))-min(test_X(:,i)));
end
%% SVM %%
% train SVM with linear model
C = [10e-7 10e-6 10e-5 10e-4 10e-3 10e-2 10e-1 10e1 10e2 10e3];
linear_model_c_value = zeros(size(C,2),1);
    
for i=1:size(C,2)
    linear_model_c_value(i) = svmtrain(train_Y, train_X, sprintf('-c %d -t 0 -v 5',C(i)));
end
    
[~,C_index] = max(linear_model_c_value);
highest_cv_C_value = C(C_index);
average_cv_accuracy_linearSVM = sum(linear_model_c_value)/length(linear_model_c_value);
linear_model_best_c{1} = svmtrain(train_Y, train_X, sprintf('-c %d -t 0',highest_cv_C_value));

[~,accuracy_vector,~] = svmpredict(test_Y, test_X, linear_model_best_c{1});
accuracy_linear = accuracy_vector(1);

% train SVM with polynomial degree 2 model 
poly2_model_c_g = zeros(length(C),length(G));
max_i = 1; max_j = 1; max_c_g = 0;

for i=1:length(C)
    for j=1:length(G)
        poly2_model_c_g(i,j) = svmtrain(dataTrainY, dataTrainX, sprintf('-t 1 -d 2 -v 5 -c %d -g %f',C(i),G(j)));
        if(poly2_model_c_g(i,j) > max_c_g)
            max_c_g = poly2_model_c_g(i,j);
            max_i = i;
            max_j = j;
        end
     end
end

best_C_G_Value(1,1) = C(max_i);
best_C_G_Value(1,2) = G(max_j);
CV_accuracy_poly2 = max_c_g;
average_cv_accuracy_poly2SVM = sum(sum(poly2_model_c_g))/numel(poly2_model_c_g);

poly2_model_best{1} = svmtrain(dataTrainY, dataTrainX, sprintf('-t 1 -d 2 -c %d -g %f',C(max_i),G(max_j)));
[~,accuracy_vector,~] = svmpredict(dataTestY, dataTestX, poly2_model_best{1});

svmpredict(dataTestY, dataTestX, poly2_model_best{1});
accuracy_poly2 = accuracy_vector(1);


% train SVM with polynomial degree 3 model 
poly3_model_c_g = zeros(length(C),length(G));
max_i = 1; max_j = 1; max_c_g = 0;

for i=1:length(C)
    for j=1:length(G)
        poly3_model_c_g(i,j) = svmtrain(dataTrainY, dataTrainX, sprintf('-t 1 -d 3 -v 5 -c %d -g %f',C(i),G(j)));
        if(poly3_model_c_g(i,j) > max_c_g)
            max_c_g = poly3_model_c_g(i,j);
            max_i = i;
            max_j = j;
        end
     end
end

best_C_G_Value(1,1) = C(max_i);
best_C_G_Value(1,2) = G(max_j);
CV_accuracy_poly3 = max_c_g;
average_cv_accuracy_poly3SVM = sum(sum(poly3_model_c_g))/numel(poly3_model_c_g);

poly3_model_best{1} = svmtrain(dataTrainY, dataTrainX, sprintf('-t 1 -d 3 -c %d -g %f',C(max_i),G(max_j)));
[~,accuracy_vector,~] = svmpredict(dataTestY, dataTestX, poly3_model_best{1});

svmpredict(dataTestY, dataTestX, poly3_model_best{1});
accuracy_poly3 = accuracy_vector(1);

% train SVM with RBF model
G = [0.001 0.005 0.01 0.05 0.1 0.5 1 2]
RBF_model_c_g = zeros(length(C),length(G));
max_i = 1; max_j = 1; max_c_g = 0;

for i=1:length(C)
    for j=1:length(G)
        RBF_model_c_g(i,j) = svmtrain(dataTrainY, dataTrainX, sprintf('-t 2 -v 5 -c %d -g %f',C(i),G(j)));
        if(RBF_model_c_g(i,j) > max_c_g)
            max_c_g = RBF_model_c_g(i,j);
            max_i = i;
            max_j = j;
        end
     end
end
    
C_G_Value(1,1) = C(max_i);
C_G_Value(1,2) = G(max_j);
best_CV_accuracy_RBF = max_c_g;
average_CV_accuracy_RBFSVM = sum(sum(RBF_model_c_g))/numel(RBF_model_c_g);
RBF_model_best{1} = svmtrain(dataTrainY, dataTrainX, sprintf('-t 2 -c %d -g %f',C(max_i),G(max_j)));
[~,accuracy_vector,~] = svmpredict(dataTestY, dataTestX, RBF_model_best{1});

svmpredict(dataTestY, dataTestX, RBF_model_best{1});
accuracy_RBF(1) = accuracy_vector(1);

%% Decision Tree %%
leafs = logspace(1,2,10);
rng('default');
err = zeros(numel(leafs),1);
for n=1:numel(leafs)
    t = fitctree(train_X,train_Y,'CrossVal','On','MinLeafSize',leafs(n));
    err(n) = kfoldLoss(t);
end

[minTrainErr, minIndex] = min(err);
bestLeafSize = leafs(minIndex);
plot(leafs, err);
xlabel('Min Leaf Size');
ylabel('cross-validation error');

OptimalTree = fitctree(test_X, test_Y,'MinLeafSize',bestLeafSize);
view(OptimalTree,'mode','graph');
[~,~,~,bestlevel] = cvLoss(OptimalTree,'SubTrees','All','TreeSize','min');
OptimalTree2 = prune(OptimalTree,'Level',bestlevel);
[label, score] = predict(OptimalTree2, test_X);
numMisclass = sum(label ~= test_Y);
misclass_rate = numMisclass/numel(test_Y);

% Metrics
ACC = 1 - misclass_rate;
EVAL = Evaluate(test_Y,label);
precision = EVAL(4);
recall = EVAL(5);
Fscore = EVAL(6);
Gmeasure = EVAL(7);
%% KNN %%
% using the KNN implemtation provided in MATLAB Machine learning toolbox
K = [1,193,385,577,769,960,1152,1344,1536,1728,1920,2112,2304,2496,2688,...
    2880,3072,3264,3456,3648,3840,4032,4224,4416,4608,4800,4992];

% Unweighted Euclidean distance KNN
LossCV1 = zeros(length(K),1);

for i=1:length(K)
    Mdl = fitcknn(train_X,train_Y,'NumNeighbors',K(i),'CrossVal',...
        'on','KFold',5,'Distance','euclidean','DistanceWeight','equal',...
        'BreakTies','nearest','Standardize',1);
    LossCV1(i) = kfoldLoss(Mdl);
end

[~,BestK_1_Idx] = min(LossCV1);

Optimal_Mdl1 = fitcknn(train_X,train_Y,'NumNeighbors',K(BestK_1_Idx),'CrossVal',...
        'off','BreakTies','nearest','Distance','euclidean','DistanceWeight','equal',...
        'Standardize',1);
[label1, score1] = predict(Optimal_Mdl1,test_X);
numMisclass1 = sum(label1 ~= test_Y);
misclass_rate1 = numMisclass1/numel(test_Y);

% Metrics
ACC = 1 - misclass_rate1;
EVAL = Evaluate(test_Y,label1);
precision = EVAL(4);
recall = EVAL(5);
Fscore = EVAL(6);
Gmeasure = EVAL(7);
%% Weighted Euclidean distance KNN with more emphasis on nearest neighbors
LossCV2 = zeros(length(K),1);

for i=1:length(K)
    Mdl = fitcknn(train_X,train_Y,'NumNeighbors',K(i),'CrossVal',...
        'on','KFold',5,'Distance','euclidean','DistanceWeight','inverse',...
        'BreakTies','nearest','Standardize',1);
    LossCV2(i) = kfoldLoss(Mdl);
end

[~,BestK_2_Idx] = min(LossCV2);

Optimal_Mdl2 = fitcknn(train_X,train_Y,'NumNeighbors',K(BestK_2_Idx),'CrossVal',...
        'off','Distance','euclidean','DistanceWeight','inverse',...
        'BreakTies','nearest','Standardize',1);
[label2 score2] = predict(Optimal_Mdl2,test_X);
numMisclass2 = sum(label2 ~= test_Y);
misclass_rate2 = numMisclass2/numel(test_Y);

% Metrics
ACC = 1 - misclass_rate2;
EVAL = Evaluate(test_Y,label2);
precision = EVAL(4);
recall = EVAL(5);
Fscore = EVAL(6);
Gmeasure = EVAL(7);
%% Locally Weighted Avearging KNN with kernel width betweem 2e0 2e10
% KernelW = [2] times euclidean distance.
 

w = [2e0;2e1;2e2;2e3;2e4;2e5;2e6;2e7;2e8;2e9;2e10];
localWeightedAvg = @(y,Z,wt)exp(sqrt((bsxfun(@minus,y,Z).^2))*wt);

max_i = 1; max_j = 1; max_acc_wt_k = 0;
LossCV3 = zeros(length(K),length(w));
 
for i=1:length(K)
    for j=1:length(w)
        Mdl_wt_k = fitcknn(train_X,train_Y,'Distance',@(y,Z)localWeightedAvg(y,Z,repmat(w(j),16,1)'),...
            'NumNeighbors',K(i),'Standardize',1,'BreakTies','nearest');
        Mdl_wt_k = crossval(Mdl_wt_k);
        LossCV3(i,j) = kfoldLoss(Mdl_wt_k);
        if(LossCV3(i,j) > max_wt_k)
            max_acc_wt_t = LossCV3(i,j);
            max_i = i;
            max_j = j;
        end
     end
end

best_wt_k_Value(1,1) = K(max_i);
best_wt_k_Value(1,2) = w(max_j);
CV_accuracy_KNN3 = max_acc_wt_t;

Optimal_Mdl3 = fitcknn(train_X,train_Y,'NumNeighbors',best_wt_k_Value(1,1),'CrossVal',...
        'off','BreakTies','nearest','Standardize',1,'Distance',@(x,Z)localWeightedAvg(x,Z,best_wt_k_Value(1,2)));
[label3 score3] = predict(Optimal_Mdl3,test_X);
numMisclass3 = sum(label3 ~= test_Y);
misclass_rate3 = numMisclass3/numel(test_Y);

% Metrics
ACC = 1 - misclass_rate3;
EVAL = Evaluate(test_Y,label3);
precision = EVAL(4);
recall = EVAL(5);
Fscore = EVAL(6);
Gmeasure = EVAL(7);
%% Bagged Decision Tree %%
leaf = logspace(1,2,10);

err = zeros(numel(leaf),1);

for i=1:length(leaf)
    t = TreeBagger(100,train_X,train_Y,'Method','classification','OOBPred','On',...
            'MinLeafSize',leaf(i));
    oobErr = oobError(t);
    err(i) = mean(oobErr);
end

[minTrainErr, minIndex] = min(err);
bestLeafSizeBAG = leaf(minIndex);

OptimalTree = fitctree(test_X, test_Y,'MinLeafSize',bestLeafSizeBAG);
[~,~,~,bestlevel] = cvLoss(OptimalTree,'SubTrees','All','TreeSize','min');

OptimalTree2 = prune(OptimalTree,'Level',bestlevel);%%%%
OptimalTree2 = OptimalTree2.compact;
[predClass,score] = predict(OptimalTree2, test_X);
numMisclass = sum(predClass ~= test_Y);
misclass_rate = numMisclass/numel(test_Y);

% Metrics
ACC = 1 - misclass_rate;
EVAL = Evaluate(test_Y,predClass);
precision = EVAL(4);
recall = EVAL(5);
Fscore = EVAL(6);
Gmeasure = EVAL(7);
%% Naive Bayes %%
err = zeros(3,1);
% using a gaussian kernel
NB_Mdl1 = fitcnb(train_X,train_Y,'Distribution','kernel','Kernel','normal','CrossVal','On');
err(1) = kfoldLoss(NB_Mdl1,'folds',5);

% using uniform kernel
NB_Mdl2 = fitcnb(train_X,train_Y,'Distribution','kernel','Kernel','box','CrossVal','On');
err(2) = kfoldLoss(NB_Mdl2,'folds',5);

% using epanechnikov kernel
NB_Mdl3 = fitcnb(train_X,train_Y,'Distribution','kernel','Kernel','epanechnikov','CrossVal','On');
err(3) = kfoldLoss(NB_Mdl3,'folds',5);

[minCVErr, minIndex] = min(err);

%%
optimalNB = fitcnb(train_X,train_Y,'Distribution','kernel','Kernel','normal');
[label,score,~] = predict(optimalNB,test_X);
numMisclass = sum(label ~= test_Y);
misclass_rate = numMisclass/numel(test_Y);

% Metrics
ACC = 1 - misclass_rate;
EVAL = Evaluate(test_Y,label);
precision = EVAL(4);
recall = EVAL(5);
Fscore = EVAL(6);
Gmeasure = EVAL(7);

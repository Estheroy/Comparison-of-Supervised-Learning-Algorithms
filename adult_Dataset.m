%% Adult dataset
%%%%%%%%%%%%%%%%%%%%%% Preprocessing Adult dataset %%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

fid=fopen('adult.txt');
data = textscan(fid,'%u %s %u %s %u %s %s %s %s %s %u %u %u %s %s','Delimiter',',');
fclose(fid);

% convert all the categorical data to numerical
data{1,2}(strcmp(data{1,2},'Private')) = {1};
data{1,2}(strcmp(data{1,2},'Self-emp-not-inc')) = {2};
data{1,2}(strcmp(data{1,2},'Self-emp-inc')) = {3};
data{1,2}(strcmp(data{1,2},'Federal-gov')) = {4};
data{1,2}(strcmp(data{1,2},'Local-gov')) = {5};
data{1,2}(strcmp(data{1,2},'State-gov')) = {6};
data{1,2}(strcmp(data{1,2},'Without-pay')) = {7};
data{1,2}(strcmp(data{1,2},'Never-worked')) = {8};
% replace the missing data ? by the median of the dataset
distri = [];
for i=1:length(data{1,2})
    if(~iscellstr(data{1,2}(i)))
        distri = [distri cell2mat(data{1,2}(i))];
    end
end
medians = median(distri);
data{1,2}(strcmp(data{1,2},'?')) = {medians};



% convert all the categorical data to numerical
data{1,4}(strcmp(data{1,4},'Bachelors')) = {1};
data{1,4}(strcmp(data{1,4},'Some-college')) = {2};
data{1,4}(strcmp(data{1,4},'11th')) = {3};
data{1,4}(strcmp(data{1,4},'HS-grad')) = {4};
data{1,4}(strcmp(data{1,4},'Prof-school')) = {5};
data{1,4}(strcmp(data{1,4},'Assoc-acdm')) = {6};
data{1,4}(strcmp(data{1,4},'Assoc-voc')) = {7};
data{1,4}(strcmp(data{1,4},'9th')) = {8};
data{1,4}(strcmp(data{1,4},'7th-8th')) = {9};
data{1,4}(strcmp(data{1,4},'12th')) = {10};
data{1,4}(strcmp(data{1,4},'Masters')) = {11};
data{1,4}(strcmp(data{1,4},'1st-4th')) = {12};
data{1,4}(strcmp(data{1,4},'10th')) = {13};
data{1,4}(strcmp(data{1,4},'Doctorate')) = {14};
data{1,4}(strcmp(data{1,4},'5th-6th')) = {15};
data{1,4}(strcmp(data{1,4},'Preschool')) = {16};

% replace the missing data ? by the median of the dataset
distri = [];
for i=1:length(data{1,4})
    if(~iscellstr(data{1,4}(i)))
        distri = [distri cell2mat(data{1,4}(i))];
    end
end
medians = median(distri);
data{1,4}(strcmp(data{1,4},'?')) = {medians};

% convert all the categorical data to numerical
data{1,6}(strcmp(data{1,6},'Married-civ-spouse')) = {1};
data{1,6}(strcmp(data{1,6},'Divorced')) = {2};
data{1,6}(strcmp(data{1,6},'Never-married')) = {3};
data{1,6}(strcmp(data{1,6},'Separated')) = {4};
data{1,6}(strcmp(data{1,6},'Widowed')) = {5};
data{1,6}(strcmp(data{1,6},'Married-spouse-absent')) = {6};
data{1,6}(strcmp(data{1,6},'Married-AF-spouse')) = {7};

% replace the missing data ? by the median of the dataset
distri = [];
for i=1:length(data{1,6})
    if(~iscellstr(data{1,6}(i)))
        distri = [distri cell2mat(data{1,6}(i))];
    end
end
medians = median(distri);
data{1,6}(strcmp(data{1,6},'?')) = {medians};

% convert all the categorical data to numerical
data{1,7}(strcmp(data{1,7},'Tech-support')) = {1};
data{1,7}(strcmp(data{1,7},'Craft-repair')) = {2};
data{1,7}(strcmp(data{1,7},'Other-service')) = {3};
data{1,7}(strcmp(data{1,7},'Sales')) = {4};
data{1,7}(strcmp(data{1,7},'Exec-managerial')) = {5};
data{1,7}(strcmp(data{1,7},'Prof-specialty')) = {6};
data{1,7}(strcmp(data{1,7},'Handlers-cleaners')) = {7};
data{1,7}(strcmp(data{1,7},'Machine-op-inspct')) = {8};
data{1,7}(strcmp(data{1,7},'Adm-clerical')) = {9};
data{1,7}(strcmp(data{1,7},'Farming-fishing')) = {10};
data{1,7}(strcmp(data{1,7},'Transport-moving')) = {11};
data{1,7}(strcmp(data{1,7},'Priv-house-serv')) = {12};
data{1,7}(strcmp(data{1,7},'Protective-serv')) = {13};
data{1,7}(strcmp(data{1,7},'Armed-Forces')) = {14};
% replace the missing data ? by the median of the dataset
distri = [];
for i=1:length(data{1,7})
    if(~iscellstr(data{1,7}(i)))
        distri = [distri cell2mat(data{1,7}(i))];
    end
end
medians = median(distri);
data{1,7}(strcmp(data{1,7},'?')) = {medians};

% convert all the categorical data to numerical
data{1,8}(strcmp(data{1,8},'Wife')) = {1};
data{1,8}(strcmp(data{1,8},'Own-child')) = {2};
data{1,8}(strcmp(data{1,8},'Husband')) = {3};
data{1,8}(strcmp(data{1,8},'Not-in-family')) = {4};
data{1,8}(strcmp(data{1,8},'Other-relative')) = {5};
data{1,8}(strcmp(data{1,8},'Unmarried')) = {6};
% replace the missing data ? by the median of the dataset
distri = [];
for i=1:length(data{1,8})
    if(~iscellstr(data{1,8}(i)))
        distri = [distri cell2mat(data{1,8}(i))];
    end
end
medians = median(distri);
data{1,8}(strcmp(data{1,8},'?')) = {medians};

% convert all the categorical data to numerical
data{1,9}(strcmp(data{1,9},'White')) = {1};
data{1,9}(strcmp(data{1,9},'Asian-Pac-Islander')) = {2};
data{1,9}(strcmp(data{1,9},'Amer-Indian-Eskimo')) = {3};
data{1,9}(strcmp(data{1,9},'Other')) = {4};
data{1,9}(strcmp(data{1,9},'Black')) = {5};
% replace the missing data ? by the median of the dataset
distri = [];
for i=1:length(data{1,9})
    if(~iscellstr(data{1,9}(i)))
        distri = [distri cell2mat(data{1,9}(i))];
    end
end
medians = median(distri);
data{1,9}(strcmp(data{1,9},'?')) = {medians};

% convert all the categorical data to numerical
data{1,10}(strcmp(data{1,10},'Female')) = {1};
data{1,10}(strcmp(data{1,10},'Male')) = {2};
% replace the missing data ? by the median of the dataset
distri = [];
for i=1:length(data{1,10})
    if(~iscellstr(data{1,10}(i)))
        distri = [distri cell2mat(data{1,10}(i))];
    end
end
medians = median(distri);
data{1,10}(strcmp(data{1,10},'?')) = {medians};

% convert all the categorical data to numerical
data{1,14}(strcmp(data{1,14},'United-States')) = {1};
data{1,14}(strcmp(data{1,14},'Cambodia')) = {2};
data{1,14}(strcmp(data{1,14},'England')) = {3};
data{1,14}(strcmp(data{1,14},'Puerto-Rico')) = {4};
data{1,14}(strcmp(data{1,14},'Canada')) = {5};
data{1,14}(strcmp(data{1,14},'Germany')) = {6};
data{1,14}(strcmp(data{1,14},'Outlying-US(Guam-USVI-etc)')) = {7};
data{1,14}(strcmp(data{1,14},'India')) = {8};
data{1,14}(strcmp(data{1,14},'Japan')) = {9};
data{1,14}(strcmp(data{1,14},'Greece')) = {10};
data{1,14}(strcmp(data{1,14},'South')) = {11};
data{1,14}(strcmp(data{1,14},'China')) = {12};
data{1,14}(strcmp(data{1,14},'Cuba')) = {13};
data{1,14}(strcmp(data{1,14},'Iran')) = {14};
data{1,14}(strcmp(data{1,14},'Honduras')) = {15};
data{1,14}(strcmp(data{1,14},'Philippines')) = {16};
data{1,14}(strcmp(data{1,14},'Italy')) = {17};
data{1,14}(strcmp(data{1,14},'Poland')) = {18};
data{1,14}(strcmp(data{1,14},'Jamaica')) = {19};
data{1,14}(strcmp(data{1,14},'Vietnam')) = {20};
data{1,14}(strcmp(data{1,14},'Mexico')) = {21};
data{1,14}(strcmp(data{1,14},'Portugal')) = {22};
data{1,14}(strcmp(data{1,14},'Ireland')) = {23};
data{1,14}(strcmp(data{1,14},'France')) = {24};
data{1,14}(strcmp(data{1,14},'Dominican-Republic')) = {25};
data{1,14}(strcmp(data{1,14},'Laos')) = {26};
data{1,14}(strcmp(data{1,14},'Ecuador')) = {27};
data{1,14}(strcmp(data{1,14},'Taiwan')) = {28};
data{1,14}(strcmp(data{1,14},'Haiti')) = {29};
data{1,14}(strcmp(data{1,14},'Columbia')) = {30};
data{1,14}(strcmp(data{1,14},'Hungary')) = {31};
data{1,14}(strcmp(data{1,14},'Guatemala')) = {32};
data{1,14}(strcmp(data{1,14},'Nicaragua')) = {33};
data{1,14}(strcmp(data{1,14},'Scotland')) = {34};
data{1,14}(strcmp(data{1,14},'Thailand')) = {35};
data{1,14}(strcmp(data{1,14},'Yugoslavia')) = {36};
data{1,14}(strcmp(data{1,14},'El-Salvador')) = {37};
data{1,14}(strcmp(data{1,14},'Trinadad&Tobago')) = {38};
data{1,14}(strcmp(data{1,14},'Peru')) = {39};
data{1,14}(strcmp(data{1,14},'Hong')) = {40};
data{1,14}(strcmp(data{1,14},'Holand-Netherlands')) = {41};
% replace the missing data ? by the median of the dataset
distri = [];
for i=1:length(data{1,14})
    if(~iscellstr(data{1,14}(i)))
        distri = [distri cell2mat(data{1,14}(i))];
    end
end
medians = median(distri);
data{1,14}(strcmp(data{1,14},'?')) = {medians};

% convert all the categorical data to numerical
data{1,15}(strcmp(data{1,15},'>50K')) = {1};
data{1,15}(strcmp(data{1,15},'<=50K')) = {-1};
% no missing data for this feature

%%%%%%%%%%%% Finish converting all the data to numerical data %%%%%%%%%%%%%
adult_dataset = [];
string_index = [2 4 6 7 8 9 10 14 15];
for i=1:15
    if(any(i == string_index) ~= 0)
        adult_dataset = horzcat(adult_dataset, cell2mat(data{1,i})); 
    else 
        adult_dataset = horzcat(adult_dataset, data{1,i});
    end
end

adult_datasetX = adult_dataset(:,1:14);
adult_datasetY = adult_dataset(:,15);
%%%%%%%%%%%% Finish preprocessing all the data for adult %%%%%%%%%%%%%%%%%%

%%%%%%% Start seperating training data and testing data for adult %%%%%%%%%
[trainInd, ~, testInd] = dividerand(size(adult_datasetY,1),0.15355794,0,1-0.15355794);
train_X = double(adult_datasetX(trainInd,:));
train_Y = double(adult_datasetY(trainInd,:));
test_X = double(adult_datasetX(testInd,:));
test_Y = double(adult_datasetY(testInd,:));
%%%%%%% Finish seperating training data and testing data for adult %%%%%%%%

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
    linear_model_c_value(i) = svmtrain(train_Y, train_X, sprintf('-c %d -t 0 -v 5 -h 0',C(i)));
end
    
[~,C_index] = max(linear_model_c_value);
highest_cv_C_value = C(C_index);
average_cv_accuracy_linearSVM = sum(linear_model_c_value)/length(linear_model_c_value);
linear_model_best_c{1} = svmtrain(train_Y, train_X, sprintf('-c %d -t 0',highest_cv_C_value));

[label1,accuracy_vector,~] = svmpredict(test_Y, test_X, linear_model_best_c{1});
accuracy_linear = accuracy_vector(1);


% Metrics
ACC = accuracy_linear/100;
EVAL = Evaluate(test_Y,label1);
precision = EVAL(4);
recall = EVAL(5);
Fscore = EVAL(6);
Gmeasure = EVAL(7);
%%
% train SVM with polynomial degree 2 model 
G = [0.001 0.005 0.01 0.05 0.1 0.5 1 2];
poly2_model_c_g = zeros(length(C),length(G));
max_i = 1; max_j = 1; max_c_g = 0;

for i=1:length(C)
    for j=1:length(G)
        poly2_model_c_g(i,j) = svmtrain(train_Y, train_X, sprintf('-h 0 -t 1 -d 2 -v 5 -c %d -g %f',C(i),G(j)));
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

poly2_model_best{1} = svmtrain(train_Y, train_X, sprintf('-t 1 -d 2 -c %d -g %f',C(max_i),G(max_j)));
[label2,accuracy_vector,~] = svmpredict(test_Y, test_X, poly2_model_best{1});

accuracy_poly2 = accuracy_vector(1);

% Metrics
ACC = accuracy_poly2/100;
EVAL = Evaluate(test_Y,label2);
precision = EVAL(4);
recall = EVAL(5);
Fscore = EVAL(6);
Gmeasure = EVAL(7);
%%
% train SVM with polynomial degree 3 model 
poly3_model_c_g = zeros(length(C),length(G));
max_i = 1; max_j = 1; max_c_g = 0;

for i=1:length(C)
    for j=1:length(G)
        poly3_model_c_g(i,j) = svmtrain(train_Y, train_X, sprintf('-h 0 -t 1 -d 3 -v 5 -c %d -g %f',C(i),G(j)));
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

poly3_model_best{1} = svmtrain(train_Y, train_X, sprintf('-t 1 -d 3 -c %d -g %f',C(max_i),G(max_j)));
[label3,accuracy_vector,~] = svmpredict(test_Y, test_X, poly3_model_best{1});

accuracy_poly3 = accuracy_vector(1);

% Metrics
ACC = accuracy_poly3/100;
EVAL = Evaluate(test_Y,label3);
precision = EVAL(4);
recall = EVAL(5);
Fscore = EVAL(6);
Gmeasure = EVAL(7);

%%
% train SVM with RBF model
G = [0.001 0.005 0.01 0.05 0.1 0.5 1 2];
RBF_model_c_g = zeros(length(C),length(G));
max_i = 1; max_j = 1; max_c_g = 0;

for i=1:length(C)
    for j=1:length(G)
        RBF_model_c_g(i,j) = svmtrain(train_Y, train_X, sprintf('-h 0 -t 2 -v 5 -c %d -g %f',C(i),G(j)));
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
RBF_model_best{1} = svmtrain(train_Y, train_X, sprintf('-t 2 -c %d -g %f',C(max_i),G(max_j)));
[label,accuracy_vector,~] = svmpredict(test_Y, test_X, RBF_model_best{1});

accuracy_RBF = accuracy_vector(1);

% Metrics
ACC = accuracy_RBF/100;
EVAL = Evaluate(test_Y,label);
precision = EVAL(4);
recall = EVAL(5);
Fscore = EVAL(6);
Gmeasure = EVAL(7);


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
bestLeafSizeBAG = leafs(minIndex);

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

optimalNB = fitcnb(train_X,train_Y,'Distribution','kernel','Kernel','epanechnikov');
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
clc;clear;close all;	
load('R4.mat')	
random_seed=G_out_data.random_seed ;  %界面设置的种子数 	
rng(random_seed)  %固定随机数种子 	
	
data_str=G_out_data.data_path_str ;  %读取数据的路径 	
	
	
data1=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data2=data1(:,2:end); 	
data=table2array(data1(:,2:end));	
data_biao=data2.Properties.VariableNames;  %数据特征的名称	
str_label=0; %标记输出是否字符类型	
 A_data1=data;	
 data_biao1=data_biao;	
 select_feature_num=G_out_data.select_feature_num1;   %特征选择的个数	
	
data_select=A_data1;	
feature_need_last=1:size(A_data1,2)-1;	
	
	
	
%% 数据划分	
x_feature_label=data_select(:,1:end-1);    %x特征	
y_feature_label=data_select(:,end);          %y标签	
index_label1=randperm(size(x_feature_label,1));	
index_label=G_out_data.spilt_label_data;  % 数据索引	
if isempty(index_label)	
     index_label=index_label1;	
end	
spilt_ri=G_out_data.spilt_rio;  %划分比例 训练集:验证集:测试集	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          %训练集个数	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
 %训练集，验证集，测试集	
train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
%Zscore 标准化	
%训练集	
x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
%验证集	
vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    %验证数据标准化	
vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  %验证数据标准化	
%测试集	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化	
test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 测试数据标准化  	
	
%% 参数设置	
num_pop=G_out_data.num_pop1;   %种群数量	
num_iter=G_out_data.num_iter1;   %种群迭代数	
method_mti=G_out_data.method_mti1;   %优化方法	
BO_iter=G_out_data.BO_iter;   %贝叶斯迭代次数	
min_batchsize=G_out_data.min_batchsize;   %batchsize	
max_epoch=G_out_data.max_epoch1;   %maxepoch	
hidden_size=G_out_data.hidden_size1;   %hidden_size	
attention_label=G_out_data.attention_label;   %注意力机制标签	
attention_head=G_out_data.attention_head;   %注意力机制设置	
	
	
	
%% 算法处理块	
	
	
	
hidden_size=G_out_data.hidden_size1;    %神经网络隐藏层	
disp('CNN-LSTM分类')	
t1=clock; 	
max_epoch=G_out_data.max_epoch1;    %神经网络隐藏层	
	
p_train1=reshape(train_x_feature_label_norm',size(train_x_feature_label,2),1,1,size(train_x_feature_label,1));	
	
p_vaild1=reshape(vaild_x_feature_label_norm',size(vaild_x_feature_label,2),1,1,size(vaild_x_feature_label,1));	
  	
	
p_test1=reshape(test_x_feature_label_norm',size(test_x_feature_label,2),1,1,size(test_x_feature_label,1));	
layers = [             % 建立输入层   	
  imageInputLayer([ size(train_x_feature_label,2) 1 1])%%2D-CNN         	
 convolution2dLayer([2,1],hidden_size(1))	
  batchNormalizationLayer   	
  reluLayer   	
	
  maxPooling2dLayer([2 1],'Stride',1)	
  flattenLayer 	
  lstmLayer(hidden_size(2), 'OutputMode', 'last')   	
  reluLayer  	
  fullyConnectedLayer(length(unique(train_y_feature_label)))          % 全连接层	
  softmaxLayer	
  classificationLayer];	
	
	
 options = trainingOptions('adam', ...	
  'Shuffle','every-epoch',...	
  'MaxEpochs',max_epoch, ...,	
  'MiniBatchSize',min_batchsize,...	
  'InitialLearnRate',0.001,...	
  'ValidationFrequency',20, ...	
  'Plots','training-progress');	
  [Mdl,Loss]  = trainNetwork(p_train1, categorical(train_y_feature_label), layers, options);	
 y_train_predict = double(classify(Mdl, p_train1));	
 y_vaild_predict =  double(classify(Mdl, p_vaild1));	
 y_test_predict =  double(classify(Mdl, p_test1));	
 t2=clock;	
 Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));	
	
	
figure	
subplot(2, 1, 1)	
plot(1 : length(Loss.TrainingAccuracy), Loss.TrainingAccuracy, '-', 'LineWidth', 1)	
xlabel('迭代次数'); ylabel('准确率');legend('训练集准确率');title ('训练集准确率迭代曲线');grid;set(gcf,'color','w')	
	
subplot(2, 1, 2)	
plot(1 : length(Loss.TrainingLoss), Loss.TrainingLoss, '-', 'LineWidth', 1)	
xlabel('迭代次数');ylabel('损失函数');legend('训练集损失值');title ('训练集损失函数曲线');grid;set(gcf,'color','w');	
	
	
disp(['运行时长: ',num2str(Time)])	
confMat_train = confusionmat(train_y_feature_label,y_train_predict);	
TP_train = diag(confMat_train);      TP_train=TP_train'; % 被正确分类的正样本 True Positives	
FP_train = sum(confMat_train, 1)  - TP_train;  %被错误分类的正样本 False Positives	
FN_train = sum(confMat_train, 2)' - TP_train;  % 被错误分类的负样本 False Negatives	
TN_train = sum(confMat_train(:))  - (TP_train + FP_train + FN_train);  % 被正确分类的负样本 True Negatives	
	
disp('训练集*******************************************************************************')	
accuracy_train = sum(TP_train) / sum(confMat_train(:)); accuracy_train(isnan(accuracy_train))=0; disp(['训练集accuracy：',num2str(mean(accuracy_train))])% Accuracy 	
precision_train = TP_train ./ (TP_train + FP_train); precision_train(isnan(precision_train))=0; disp(['训练集precision_train：',num2str(mean(precision_train))]) % Precision	
recall_train = TP_train ./ (TP_train + FN_train);recall_train(isnan(recall_train))=0; disp(['训练集recall_train：',num2str(mean(recall_train))])  % Recall / Sensitivity	
F1_score_train = 2 * (precision_train .* recall_train) ./ (precision_train + recall_train); F1_score_train(isnan(F1_score_train))=0;  disp(['训练集F1_score_train：',num2str(mean(F1_score_train))])   % F1 Score	
specificity_train = TN_train ./ (TN_train + FP_train); specificity_train(isnan(specificity_train))=0; disp(['训练集specificity_train：',num2str(mean(specificity_train))])  % Specificity	
	
disp('验证集********************************************************************************')	
confMat_vaild = confusionmat(vaild_y_feature_label,y_vaild_predict);	
TP_vaild = diag(confMat_vaild);      TP_vaild=TP_vaild'; % 被正确分类的正样本 True Positives	
FP_vaild = sum(confMat_vaild, 1)  - TP_vaild;  %被错误分类的正样本 False Positives	
FN_vaild = sum(confMat_vaild, 2)' - TP_vaild;  % 被错误分类的负样本 False Negatives	
TN_vaild = sum(confMat_vaild(:))  - (TP_vaild + FP_vaild + FN_vaild);  % 被正确分类的负样本 True Negatives	
accuracy_vaild = sum(TP_vaild) / sum(confMat_vaild(:)); accuracy_vaild(isnan(accuracy_vaild))=0; disp(['验证集accuracy：',num2str(accuracy_vaild)])% Accuracy 	
precision_vaild = TP_vaild ./ (TP_vaild + FP_vaild); precision_vaild(isnan(precision_vaild))=0; disp(['验证集precision_vaild：',num2str(mean(precision_vaild))]) % Precision	
recall_vaild = TP_vaild ./ (TP_vaild + FN_vaild); recall_vaild(isnan(recall_vaild))=0;  disp(['验证集recall_vaild：',num2str(mean(recall_vaild))])  % Recall / Sensitivity	
F1_score_vaild = 2 * (precision_vaild .* recall_vaild) ./ (precision_vaild + recall_vaild);  F1_score_vaild(isnan(F1_score_vaild))=0;  disp(['验证集F1_score_vaild：',num2str(mean(F1_score_vaild))])   % F1 Score	
specificity_vaild = TN_vaild ./ (TN_vaild + FP_vaild); specificity_vaild(isnan(specificity_vaild))=0; disp(['验证集specificity_vaild：',num2str(mean(specificity_vaild))])  % Specificity	
disp('测试集********************************************************************************') 	
confMat_test = confusionmat(test_y_feature_label,y_test_predict);	
TP_test = diag(confMat_test);      TP_test=TP_test'; % 被正确分类的正样本 True Positives	
FP_test = sum(confMat_test, 1)  - TP_test;  %被错误分类的正样本 False Positives	
FN_test = sum(confMat_test, 2)' - TP_test;  % 被错误分类的负样本 False Negatives	
TN_test = sum(confMat_test(:))  - (TP_test + FP_test + FN_test);  % 被正确分类的负样本 True Negatives	
	
accuracy_test = sum(TP_test) / sum(confMat_test(:)); accuracy_test(isnan(accuracy_test))=0; disp(['测试集accuracy：',num2str(accuracy_test)])% Accuracy	
precision_test = TP_test ./ (TP_test + FP_test);  precision_test(isnan(precision_test))=0; disp(['测试集precision_test：',num2str(mean(precision_test))]) % Precision	
recall_test = TP_test ./ (TP_test + FN_test); recall_test(isnan(recall_test))=0; disp(['测试集recall_test：',num2str(mean(recall_test))])  % Recall / Sensitivity	
F1_score_test = 2 * (precision_test .* recall_test) ./ (precision_test + recall_test); F1_score_test(isnan(F1_score_test))=0; disp(['测试集F1_score_test：',num2str(mean(F1_score_test))])   % F1 Score	
specificity_test = TN_test ./ (TN_test + FP_test); specificity_test(isnan(specificity_test))=0; disp(['测试集specificity_test：',num2str(mean(specificity_test))])  % Specificity	
	
	
	
%% K折验证	
x_feature_label_norm_all=(x_feature_label-x_mu)./x_sig;    %x特征	
y_feature_label_norm_all=y_feature_label;	
Kfold_num=G_out_data.Kfold_num;	
cv = cvpartition(size(x_feature_label_norm_all, 1), 'KFold', Kfold_num); % Split into K folds	
for k = 1:Kfold_num	
    trainingIdx = training(cv, k);	
    validationIdx = test(cv, k);	
     x_feature_label_norm_all_traink=x_feature_label_norm_all(trainingIdx,:);	
   y_feature_label_norm_all_traink=y_feature_label_norm_all(trainingIdx,:);	
	
   x_feature_label_norm_all_testk=x_feature_label_norm_all(validationIdx,:);	
   y_feature_label_norm_all_testk=y_feature_label_norm_all(validationIdx,:);	
	
   p_traink1=[];p_testk1=[];	
	
  	
   p_traink1=reshape(x_feature_label_norm_all_traink',size(x_feature_label_norm_all_traink,2),1,1,size(x_feature_label_norm_all_traink,1));	
   	
   p_testk1=reshape(x_feature_label_norm_all_testk',size(x_feature_label_norm_all_testk,2),1,1,size(x_feature_label_norm_all_testk,1));	
  	
 	
	
	
   optionsk = trainingOptions('adam', ... 	
        'Shuffle','every-epoch',...	
        'MaxEpochs',max_epoch, ..., 	
        'MiniBatchSize',min_batchsize,... 	
        'InitialLearnRate',0.001,... 	
        'ValidationFrequency',20);	
  	
   Mdlkf=trainNetwork(p_traink1, categorical(y_feature_label_norm_all_traink), layers, optionsk); 	
  	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
   Mdl_kfold{1,k}=Mdlkf;	
   y_test_predict_all_testk=double(classify(Mdlkf, p_testk1));  %测试集预测结果	
  	
   	
   test_kfold=sum((y_test_predict_all_testk==y_feature_label_norm_all_testk))/length(y_feature_label_norm_all_testk);% 采用的MAE	
   AUC_kfold(k)=test_kfold;	
	
	
	
end	
	
% k折验证结果绘图	
figure('color',[1 1 1]);	
	
color_set=[0.4353    0.5137    0.7490];	
plot(1:length(AUC_kfold),AUC_kfold,'--p','color',color_set,'Linewidth',1.3,'MarkerSize',6,'MarkerFaceColor',color_set,'MarkerFaceColor',[0.3,0.4,0.5]);	
grid on;	
box off;	
grid off;	
ylim([0.92*min(AUC_kfold),1.2*max(AUC_kfold)])	
xlabel('kfoldnum')	
ylabel('accuracy')	
xticks(1:length(AUC_kfold))	
set(gca,'Xgrid','off');	
set(gca,'Linewidth',1);	
set(gca,'TickDir', 'out', 'TickLength', [.005 .005], 'XMinorTick', 'off', 'YMinorTick', 'off');	
yline(mean(AUC_kfold),'--')	
%小窗口柱状图的绘制	
axes('Position',[0.6,0.65,0.25,0.25],'box','on'); % 生成子图	
GO = bar(1:length(AUC_kfold),AUC_kfold,1,'EdgeColor','k');	
GO(1).FaceColor = color_set;	
xticks(1:length(AUC_kfold))	
xlabel('kfoldnum')	
ylabel('accuracy')	
disp('****************************************************************************************') 	
disp([num2str(Kfold_num),'折验证预测准确率accuracy结果：'])	
disp(AUC_kfold) 	
disp([num2str(Kfold_num),'折验证  ','accuracy均值为： ' ,num2str(mean(AUC_kfold)),'    accuracy标准差为： ' ,num2str(std(AUC_kfold))]) 	

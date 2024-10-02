clc;clear;close all;	
load('R3.mat')	
random_seed=G_out_data.random_seed ;   	
rng(random_seed)   	
	
data_str=G_out_data.data_path_str ;   	
	
	
data1=readtable(data_str,'VariableNamingRule','preserve');  	
data2=data1(:,2:end); 	
data=table2array(data1(:,2:end));	
data_biao=data2.Properties.VariableNames;  	
str_label=0; 	
 A_data1=data;	
 data_biao1=data_biao;	
 select_feature_num=G_out_data.select_feature_num1;  	
	
data_select=A_data1;	
feature_need_last=1:size(A_data1,2)-1;	
	
x_feature_label=data_select(:,1:end-1);    	
y_feature_label=data_select(:,end);          	
index_label1=randperm(size(x_feature_label,1));	
index_label=G_out_data.spilt_label_data;  	
if isempty(index_label)	
     index_label=index_label1;	
end	
spilt_ri=G_out_data.spilt_rio;  	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); 	
 	
train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
	
x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    	
y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;     	
	
vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    
vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  	
	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    	
test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;      	
	
	
num_pop=G_out_data.num_pop1;  	
num_iter=G_out_data.num_iter1;   
method_mti=G_out_data.method_mti1;   	
BO_iter=G_out_data.BO_iter;   	
min_batchsize=G_out_data.min_batchsize;   	
max_epoch=G_out_data.max_epoch1;   	
hidden_size=G_out_data.hidden_size1;   	
	
	
	
	

	
 hidden_size=G_out_data.hidden_size1;    
disp('CNN')	
t1=clock; 	
 max_epoch=G_out_data.max_epoch1;    
	
p_train1=reshape(train_x_feature_label_norm',size(train_x_feature_label,2),1,1,size(train_x_feature_label,1));	
	
p_vaild1=reshape(vaild_x_feature_label_norm',size(vaild_x_feature_label,2),1,1,size(vaild_x_feature_label,1));	
  	
	
p_test1=reshape(test_x_feature_label_norm',size(test_x_feature_label,2),1,1,size(test_x_feature_label,1));	
layers = [               	
  imageInputLayer([ size(train_x_feature_label,2) 1 1])        	
  convolution2dLayer([2,1],hidden_size(1))	
  batchNormalizationLayer   	
  reluLayer   	
	
  maxPooling2dLayer([2 1],'Stride',1)	
  flattenLayer 	
	
  reluLayer  	
  fullyConnectedLayer(length(unique(train_y_feature_label)))          
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
	
	
	
disp(['time: ',num2str(Time)])	
confMat_train = confusionmat(train_y_feature_label,y_train_predict);	
TP_train = diag(confMat_train);      TP_train=TP_train'; 	
FP_train = sum(confMat_train, 1)  - TP_train;  
FN_train = sum(confMat_train, 2)' - TP_train;  	
TN_train = sum(confMat_train(:))  - (TP_train + FP_train + FN_train);  
	
disp('train*******************************************************************************')	
accuracy_train = sum(TP_train) / sum(confMat_train(:)); accuracy_train(isnan(accuracy_train))=0; disp(['train accuracy：',num2str(mean(accuracy_train))])	

	
disp('valid********************************************************************************')	
confMat_vaild = confusionmat(vaild_y_feature_label,y_vaild_predict);	
TP_vaild = diag(confMat_vaild);      TP_vaild=TP_vaild'; 
FP_vaild = sum(confMat_vaild, 1)  - TP_vaild;  	
FN_vaild = sum(confMat_vaild, 2)' - TP_vaild;  
TN_vaild = sum(confMat_vaild(:))  - (TP_vaild + FP_vaild + FN_vaild);  
accuracy_vaild = sum(TP_vaild) / sum(confMat_vaild(:)); accuracy_vaild(isnan(accuracy_vaild))=0; disp(['vaild accuracy：',num2str(accuracy_vaild)]) 	
	
disp('test********************************************************************************') 	
confMat_test = confusionmat(test_y_feature_label,y_test_predict);	
TP_test = diag(confMat_test);      TP_test=TP_test'; 
FP_test = sum(confMat_test, 1)  - TP_test; 
FN_test = sum(confMat_test, 2)' - TP_test; 	
TN_test = sum(confMat_test(:))  - (TP_test + FP_test + FN_test); 
	
accuracy_test = sum(TP_test) / sum(confMat_test(:)); accuracy_test(isnan(accuracy_test))=0; disp(['test accuracy：',num2str(accuracy_test)])
	
	
	
	
	

x_feature_label_norm_all=(x_feature_label-x_mu)./x_sig;    
y_feature_label_norm_all=y_feature_label;	
Kfold_num=G_out_data.Kfold_num;	
cv = cvpartition(size(x_feature_label_norm_all, 1), 'KFold', Kfold_num); 
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
  	
   Mdlkf=trainNetwork(p_traink1, y_feature_label_norm_all_traink, layers, optionsk); 	

   Mdl_kfold{1,k}=Mdlkf;	
   y_test_predict_all_testk=double(classify(Mdlkf, p_testk1));  
  	
   	
   test_kfold=sum((y_test_predict_all_testk==y_feature_label_norm_all_testk))/length(y_feature_label_norm_all_testk);
	
	
	
end	
	

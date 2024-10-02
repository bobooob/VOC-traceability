function [Mdl]  = optimizebaye_fitCTreeBagger(train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm,max_iter)    
    maxMinLS = 20;   
    tree_range=[10,300];
    tree_num = optimizableVariable('tree_num',tree_range,'Type','integer');
    minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
%     numPTS = optimizableVariable('numPTS',[1,size(train_x_feature_label_norm,2)],'Type','integer');
    hyperparametersRF = [tree_num;minLS];
    results = bayesopt(@(params)oobErrRF(params,train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0,...
    'MaxObjectiveEvaluations',max_iter);
    bestHyperparameters = results.XAtMinObjective;
  
    Mdl = TreeBagger(bestHyperparameters.tree_num,train_x_feature_label_norm,train_y_feature_label_norm,'Method','classification',...
    'MinLeafSize',bestHyperparameters.minLS);
     disp(['Bayesian', 'optimise TreeBagger:   ',"Tree_Num:",num2str(bestHyperparameters.tree_num),'   MinLeafSize: ',num2str(bestHyperparameters.minLS)]) 
end
function [oobErr] = oobErrRF(params,train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm)

    Mdl = TreeBagger(params.tree_num,train_x_feature_label_norm,train_y_feature_label_norm,'Method','classification',...
    'MinLeafSize',params.minLS);

     P_vaild_y_feature_label_norm1=predict(Mdl,vaild_x_feature_label_norm);
            for i=1:length(P_vaild_y_feature_label_norm1)
                P_vaild_y_feature_label_norm(i,1)=str2double(P_vaild_y_feature_label_norm1{i,1});
           end

    oobErr=1-sum((P_vaild_y_feature_label_norm==vaild_y_feature_label_norm))/length(vaild_y_feature_label_norm);

end
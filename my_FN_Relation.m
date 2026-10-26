% function [int_relation, relation] = my_FN_Relation(data, index)
% % % 模糊邻域相似关系(加速版）
% 
%     [row, col] = size(data);
%     col_count = col - 1;
%     relation = cell(col_count, 1);
%     
%     % Precompute the pairwise absolute differences for all features 预先计算所有特征的成对绝对差
%     diffs = zeros(row, row, col_count);
%     for k = 1:col_count
%         diffs(:,:,k) = abs(data(:,k) - data(:,k)');
%         relation{k} = 1 - diffs(:,:,k); % Initialize the relation matrices 初始化关系矩阵
%         relation{k}(sub2ind([row, row], 1:row, 1:row)) = 1;  % Set diagonal to 1 将对角线设置为 1
%     end
%     
%     % Handle the case where sample classes are the same 处理样本类相同的情况
%     for i = 1:row
%         for j = i+1:row
%             if index(i) == index(j)
%                 for k = 1:col_count
%                     relation{k}(i, j) = 1;
%                     relation{k}(j, i) = 1;
%                 end
%             end
%         end
%     end
%     
%     % Compute the intersection of all relations
%     int_relation = relation{1};
%     for k = 2:col_count
%         int_relation = min(int_relation, relation{k});
%     end
% end


function [int_relation, relation] = my_FN_Relation(data, index)
    [row, col] = size(data);
    col_count = col - 1;  % Assuming the last column is the class label
    relation = cell(col_count, 1);
    
    % Precompute the pairwise absolute differences for all features
    % Utilize MATLAB matrix operations to handle all features at once
    for k = 1:col_count
        % Compute absolute differences using broadcasting
        diffs = abs(data(:,k) - data(:,k)');
        relation{k} = 1 - diffs / max(diffs(:));  % Normalize by the maximum difference to scale between 0 and 1
        relation{k}(1:row+1:end) = 1;  % Set diagonal elements to 1
    end
    
    % Pre-compute indices where index matches, without explicit looping
    same_class = index == index';
    for k = 1:col_count
        relation{k}(same_class) = 1;  % Set to 1 where samples are of the same class
    end
    
    % Compute the intersection of all relations
    int_relation = relation{1};
    for k = 2:col_count
        int_relation = min(int_relation, relation{k});
    end
end



% function [int_relation,relation]= my_FN_Relation(data,index)
% % 模糊邻域相似关系
% 
% [row, col]=size(data);
% relation = cell(col-1,1);
% for k=1:col-1
%     relation{k}=zeros(row);
% end
% int_relation = ones(row);
% for k = 1:col-1
%     for i=1:row-1
%         for j=i+1:row
% 
%             %             Rij = 0;
%             %             for kk = 1:col-1
%             %                 Rij = Rij + (data(i,kk)-data(j,kk))^2;
%             %             end
%             %             Rij = sqrt(Rij)/(col-1);
% 
%             if index(i)==index(j) && index(i)~=0 %
%                 relation{k}(i,j)=1;
%             else
%                 relation{k}(i,j)=1-abs(data(i,k)-data(j,k));
%                 %relation{k}(i,j)=1-Rij;
%             end
% 
%         end
%     end
%     relation{k}=relation{k}+relation{k}'+eye(row);
%     int_relation = min(int_relation, relation{k});
% end
% 
% end


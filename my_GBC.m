function [C,IDX] = my_GBC(data)
%Granular ball computing 返回：生成粒球的中心、半径，样本属于哪个粒球
%改进的GBC

Centr = my_canopy( data ); % 初始中心点的选择

[row,col] = size(data); % 获取数据的行数和列数
class = data(:,end); % 获取数据的类别标签
classnum = length(unique(class)); % 计算类别的数量

% 初始化所有样本的粒球标签，初始时所有样本同属于一个粒球
label = ones(row,1);
group = unique(label); % unique 返回和label中一样的值，但是没有重复元素，结果按升序排序。返回去重后的粒球列表
no_division_group = []; % 当粒球纯度为1时，不再对粒球进行分裂,no_division_group记录此时group的值，即粒球编号

pure=0; % 纯度初始化

% 当所有粒球的纯度未达到1时循环
while pure < 1
    % 对每个粒球进行处理，进行k分类
    for i = 1:length(group)
        if ~ismember(group(i),no_division_group)
            % 找出属于当前粒球的样本
            tmp = (label == group(i));
            % tmp = bsxfun(@eq,label,group(i)); %bsxfun 调用eq（等于）函数对label和group(i)两个矩阵进行操作，label:row行1列，group(i):1行1列。eq返回一个逻辑数组（0，1）。
            % 如果当前粒球的样本数大于等于类别数
            if sum(tmp) >= classnum % 当tmp中只有一个为1或全为0时，不满足if条件。
                tmplabel = litekmeans(data(tmp,1:end-1),classnum,'Start',Centr); % 对这些样本进行kmeans聚类
                % 更新这些样本的粒球标签
                label(tmp) = max(label)+tmplabel;% label中tmp==1的位置 = label中最大元素 + 聚类后 label发生了变化，但是没有分裂的粒球label值并未发生变化
            end
        end
    end
    group = unique(label); % 更新粒球标签列表
    ball = zeros(row,1); % 初始化粒球标记
    all_pure = zeros(1,length(group)); % 初始化所有粒球的纯度记录

    % 求解粒球，计算每个粒球的中心和半径
    for i = 1:length(group)
        tmp = (label == group(i));
        % tmp = bsxfun(@eq,label,group(i)); % tmp==1表示样本属于第i个粒球。
        ball(tmp,1) = group(i);

        if ~ismember(group(i),no_division_group)
            if sum(tmp) >= classnum
                center = mean(data(tmp,1:end-1)); % 计算粒球中心
                % 计算粒球中样本的平均距离
                distance = pdist2(center,data(tmp,1:end-1)); 
                r = mean(distance);
                tmpdistance = distance <= r; % tmpdistance == 1 表示样本在粒球内部
                ball(tmp,1) = tmpdistance.* group(i); % 将属于粒球却不在粒球内部的样本剔除
                number = tmpdistance'.*class(tmp,1);
                number(number==0) = [];
                all_pure(1,i)=sum(number==mode(number))./sum(tmpdistance);
            else
                all_pure(1,i)=1;
            end

            if all_pure(1,i) == 1
                no_division_group = [no_division_group,group(i)]; % 当粒球纯度为1时，不再对粒球进行分裂,此后不再计算粒球包含的样本
            end

        else
            all_pure(1,i) = 1;
        end

    end
    pure = mean(all_pure);
end

% 筛选需要的粒球
Group = unique(ball); % ball == 0 表示样本属于粒球却不在粒球内。如果设置粒球的半径为粒球内样本距离中心最大的值，则会避免出现该情况。
Group(Group==0) = []; % 把数值为0的一行删去，如果粒球半径发生变化，则不需要此步骤。
GB_num = size(Group,1);
Center = {1,GB_num};
GB_label = zeros(1,GB_num);
for i = 1:GB_num
    tmp = (ball == Group(i));
    % tmp = bsxfun(@eq,ball,Group(i));
    if sum(tmp) >= 2  % 如果有至少两个样本属于该粒球，则需求出中心、半径。
        Center{1,i} = mean(data(tmp,1:end-1));
    else
        Center{1,i} = data(tmp,1:end-1);
    end
    GB_label(1,i) = mode(class(tmp,1));
end
Center = cell2mat(Center);
Center = reshape(Center,col-1,GB_num); % reshape函数将Center排列成col-1行GB_num列
Center = Center';

[IDX,C] = litekmeans(data(:,1:end-1),GB_num,'Start',Center); % 为ball==0的样本重新聚类

end




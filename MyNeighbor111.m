function [Xiajinsi, int_relation] = MyNeighbor111(X, Y)
    % 合并数据和标签
    data = [X, Y];
    [row, ~] = size(X);

    %% 生成粒球
    [~, index] = my_GBC(data);  % 假设 my_GBC 返回聚类索引

    %% 计算模糊邻域相似关系
    [int_relation, relation] = my_FN_Relation(data, index);

    %% 计算模糊决策
    [partition, ~, cj] = unique(Y);
    class_num = length(partition);
    
    % 预先计算每个类的成员索引
    class = arrayfun(@(x) find(cj == x), 1:class_num, 'UniformOutput', false);

    FD = zeros(row, class_num);
    for j = 1:class_num
        FD(:, j) = sum(int_relation(:, class{j}), 2) ./ sum(int_relation, 2);
    end

    %% 计算下近似
    Xiajinsi = zeros(class_num, row);

    % 使用向量化操作计算下近似
    for k = 1:class_num
        BD = max(1 - int_relation, FD(:, k));  % [row x row]
        Xiajinsi(k, :) = min(BD, [], 2)';  % 取每行的最小值，并转置为行向量
    end
end

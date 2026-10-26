function [selected_features, Xiajinsi, int_relation] = OFS_FNGBRS111(X, Y)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%基于FNGBRS的OFS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % X, Y: 更新后的数据和标签
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%基于FNGBRS的OFS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    data = [X, Y];
    [row, col] = size(X);

    %% 更新粒球结构
    [~, index] = my_GBC(data);  % 假设 my_GBC 返回聚类索引

    %% 更新模糊邻域相似关系
    [int_relation, relation] = my_FN_Relation(data, index);
    % relation 是一个 cell 数组，存储每个特征的相似度矩阵

    %% 更新模糊决策
    [partition, ~, cj] = unique(Y);
    class_num = length(partition);
    class = arrayfun(@(x) find(cj == x), 1:class_num, 'UniformOutput', false);

    FD = zeros(row, class_num);
    for i = 1:row
        sum_int_relation_i = sum(int_relation(i, :));  % 分母预计算
        for j = 1:class_num
            FD(i, j) = sum(int_relation(i, class{j})) / sum_int_relation_i;
        end
    end

    %% 更新下近似
    Xiajinsi = zeros(class_num, row);

    % 使用向量化操作计算下近似
    for k = 1:class_num
        BD = max(1 - int_relation, FD(:, k));  % [row x row]
        Xiajinsi(k, :) = min(BD, [], 2)';  % 取每行的最小值，并转置为行向量
    end

    %% 属性约简
    red = zeros(1, col);

    %% 计算模糊邻域熵
    FN_entropy = my_FN_entropy(row, int_relation, FD, class_num);

    for j = 1:col
        %% 内部重要度--模糊邻域熵/依赖度
        % 计算删除第 j 个特征后的模糊邻域关系
        fuzzy_neighbor_deletek = ones(row, row);
        for w = 1:col
            if w ~= j
                fuzzy_neighbor_deletek = min(fuzzy_neighbor_deletek, relation{w});
            end
        end

        FNE_Deletek = my_FN_entropy(row, fuzzy_neighbor_deletek, FD, class_num);
        Sig_inner = FNE_Deletek - FN_entropy;

        if Sig_inner > 0
            red(j) = 1;
        else
            red(j) = 0;
        end
    end

    %% 外部重要度
    inter_Selected = find(red == 1);
    col_middle = length(inter_Selected);
    fuzzy_neighbor_middle = ones(row, row);
    for i = 1:col_middle
        fuzzy_neighbor_middle = min(fuzzy_neighbor_middle, relation{inter_Selected(i)});
    end

    FNE_middle = my_FN_entropy(row, fuzzy_neighbor_middle, FD, class_num);

    while FNE_middle > FN_entropy
        max_Sig_outer = -inf;  % 初始化为负无穷
        max_index = -1;  % 初始化为无效值
        for j = 1:col
            if ~ismember(j, inter_Selected)
                % 计算添加第 j 个特征后的模糊邻域关系
                fuzzy_neighbor_add_J = min(fuzzy_neighbor_middle, relation{j});
                FNE_add_J = my_FN_entropy(row, fuzzy_neighbor_add_J, FD, class_num);

                Sig_outer = FNE_add_J - FNE_middle;

                if Sig_outer > max_Sig_outer
                    max_Sig_outer = Sig_outer;
                    max_index = j;
                end
            end
        end

        if max_index == -1
            break;  % 如果没有找到合适的候选特征，退出循环
        end

        red(max_index) = 1;
        inter_Selected = find(red == 1);

        % 更新模糊邻域关系
        fuzzy_neighbor_middle = min(fuzzy_neighbor_middle, relation{max_index});
        FNE_middle = my_FN_entropy(row, fuzzy_neighbor_middle, FD, class_num);
    end

    col_middle = length(inter_Selected);
    for i = 1:col_middle
        fuzzy_neighbor_rongyu = ones(row, row);
        for w = 1:col_middle
            if w ~= i
                fuzzy_neighbor_rongyu = min(fuzzy_neighbor_rongyu, relation{inter_Selected(w)});
            end
        end

        FNE_rongyu = my_FN_entropy(row, fuzzy_neighbor_rongyu, FD, class_num);

        if FNE_rongyu <= FNE_middle
           red(inter_Selected(i)) = 0;
        end
    end

    selected_features = find(red == 1);
end

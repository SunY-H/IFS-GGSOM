function clusterLabels = getClusterResults(ggsom, data)
    % 初始化聚类标签数组
    clusterLabels = zeros(size(data, 1), 1);
    
    % 遍历数据集中的每个数据点
    for i = 1:size(data, 1)
        % 获取当前数据点
        dataPoint = data(i, :);
        
        % 初始化最小距离为无穷大
        minDist = inf;
        % 初始化最佳匹配单元（BMU）索引
        bmuIndex = -1;
        
        % 遍历GSOM中的每个节点
        for j = 1:length(ggsom.nodes)
            % 计算当前数据点与节点j的欧式距离
            dist = norm(dataPoint - ggsom.nodes(j).weights);
            % dist = fuzzySimilarity(dataPoint, ggsom.nodes(j).weights);
            
            % 如果距离小于当前最小距离，则更新最小距离和BMU索引
            if dist < minDist
                minDist = dist;
                bmuIndex = j;
            end
        end
        
        % 确保找到了有效的BMU
        if bmuIndex > -1
            % 将数据点i分配给找到的BMU所代表的聚类，使用BMU的label属性
            clusterLabels(i) = ggsom.nodes(bmuIndex).label;
        else
            % 如果没有找到有效的BMU，可以赋一个默认值或进行错误处理
            clusterLabels(i) = -1; % 例如，这里使用-1作为错误标识
        end
    end
end

% function clusterLabels = getClusterResults(ggsom, data)
%     numDataPoints = size(data, 1);
%     numNodes = length(ggsom.nodes);
%     clusterLabels = zeros(numDataPoints, 1);
%     
%     % 预分配节点权重矩阵和节点标签数组
%     nodeWeights = zeros(numNodes, size(data, 2));
%     nodeLabels = zeros(numNodes, 1);
%     for j = 1:numNodes
%         nodeWeights(j, :) = ggsom.nodes(j).weights;
%         nodeLabels(j) = ggsom.nodes(j).label;
%     end
% 
%     % 并行处理每个数据点
%     parfor i = 1:numDataPoints
%         dataPoint = data(i, :);
%         
%         % 一次性计算当前数据点与所有节点的欧式距离
%         dists = sqrt(sum(bsxfun(@minus, nodeWeights, dataPoint).^2, 2));
%         
%         % 找到最小距离的索引
%         [~, bmuIndex] = min(dists);
%         
%         % 使用BMU的label属性为数据点分配聚类
%         clusterLabels(i) = nodeLabels(bmuIndex);
%     end
% end

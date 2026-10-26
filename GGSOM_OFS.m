classdef GGSOM_OFS <handle
    % GGSOM类定义了GGSOM的结构和方法

    % 以时间复杂度的提升换取空间复杂度的优化，不再增量更新模糊邻域信息粒。不再存储模糊邻域相似矩阵。

    properties
        nodes; % 存储GGSOM中的所有节点
        init_lr % 初始学习率
        initial_threshold % 初始邻域阈值
        gamma % 错误分布参数 用于控制误差累积的增加
        enableGrowth = true; % 默认启用生长
        GT % 生长阈值，用于决定何时增加新节点
        data % 输入数据集
        dim % 数据的维度
        sampleCount % 样本的个数
        Max_iteration % 最大迭代次数
        label % 用于存储节点标签
        Xiajinsi % 用于存储数据data的下近似信息
        int_relation % 用于存储数据data的模糊邻域信息粒

        testingData % 用于存储测试数据块的信息
        testingDataCompleted % 用于存储全部的测试数据块的信息

    end

    methods

        function obj = GGSOM_OFS(data, labels, spread_factor, Max_iteration, my_gamma, init_lr, init_thresh)
            % GGSOM构造函数，用于初始化GGSOM网络 初始化包括设置输入数据、计算生长阈值、创建初始节点等
            % 参数：
            % spread_factor - 扩散因子，控制网络生长的速度
            % Max_iteration - 最大迭代次数
            % my_alpha - 学习率衰减因子
            % my_gamma - 错误分布参数 用于控制误差累积的增加
            % my_threshold - 用于决定哪些神经元会被认为是获胜神经元（BMU）的邻域

            % 根据有无标签执行不同的初始化逻辑
            if nargin > 1 && ~isempty(labels)
                %% 存在标签数据，根据标签进行初始化，适用于训练阶段

                obj.data = data; % 将输入数据集赋值给对象
                obj.label = labels; % 将输入数据集的标签赋值给对象

                [obj.Xiajinsi, obj.int_relation] = MyNeighbor111(obj.data, obj.label); % Xiajinsi是下近似矩阵，每一行对应一个类别的下近似度

                centers = computeClusterCenters(obj.data, obj.Xiajinsi); % 根据Xiajinsi矩阵计算每个类别的代表中心，将其作为权重向量的初始值

                % 在计算类中心时，处理潜在的空类别或稀有类别----一般不会出现这种情况
                if any(isnan(centers))
                    warning('Some cluster centers are undefined due to empty classes.');
                    centers(isnan(centers)) = mean(obj.data); % Assign mean of all data as a fallback
                end

                % 基于标签的不同值初始化节点数量...
                % 基于标签的唯一值个数动态设置起始节点数
                unique_labels = unique(obj.label); % 获取标签的唯一值
                num_labels = length(unique_labels); % 计算标签的唯一值个数

                % 动态创建起始节点
                obj.nodes = GSOM_Node.empty(num_labels, 0);  % 预设具体数量的节点数组
                for i = 1:num_labels
                    % 为每个节点分配标签
                    nodeLabel = unique_labels(i);
                    initialWeights = centers(i, :); % 假设centers已经根据某种策略计算好了
                    LowA = obj.Xiajinsi(i,:);
                    obj.nodes(i) = GSOM_Node(nodeLabel, initialWeights, LowA);
                end

                [obj.sampleCount, obj.dim] = size(obj.data); % 确定样本个数和数据维度
                obj.GT = -obj.dim * log(spread_factor); % 计算生长阈值
                obj.Max_iteration = Max_iteration;
                % 为训练阶段和测试阶段设置不同的学习率变化公式
                obj.init_lr = init_lr;     % 学习率的初始值
                obj.initial_threshold = init_thresh; % 邻域阈值的初始值
                obj.gamma = my_gamma; % 分布因子，用于控制误差累积的增加
                obj.testingData = []; % 在训练阶段将测试数据块设置为空
                obj.testingDataCompleted = []; % 在训练阶段将测试数据块设置为空
                obj.enableGrowth = false; % 设置禁用生长的标志

            else
                %% 不存在标签数据，适用于测试阶段

                obj = getappdata(0, 'GlobalGGSOM');
                obj.testingDataCompleted = vertcat(obj.testingDataCompleted, data); % 更新测试数据块
                obj.testingData = data; % 将输入数据集赋值给对象
                obj.enableGrowth = true; % 设置启用生长的标志
                [obj.sampleCount, obj.dim] = size(obj.data); % 更新sampleCount-新的数据块流入之前样本空间中样本的个数
                obj.data = [obj.data; obj.testingData]; % 更新GGSOM中的data
                

                % 其他初始值沿用之前训练中保留的obj

            end


        end


        function  selectedFeatures = train(obj)

            % 训练GSOM网络。本方法通过遍历输入数据集，对GSOM网络进行训练，逐步调整节点权重，以匹配输入数据。

            if ~obj.enableGrowth
                %% 训练阶段包含标签，obj.enableGrowth = false; 禁用生长
                selectedFeatures = [];

            else
                %% obj.enableGrowth = true; 启用生长，根据FRS概念先提取数据块中的知识，设置下近似计算方式

                %% 先分配一个聚类结果并更新下近似

                % 如果每次迭代都对下近似进行更新，显然是费时的。另外一种方案：在检测到节点生长时才更新预测标签信息和下近似信息----这样做的前提是训练阶段对GGSOM的训练足够有效
                % 这样第一次预测得到的标签才有意义（在迭代中下近似信息不发生变化意味着神经元之间的邻域关系不发生变化）
                % 另外，对于新节点的权重、误差等初始化信息的设置就异常重要，生长节点后的下近似信息要足够具有代表性（这依赖于GGSOM对新数据块的预测精度）
                % 谨慎使用这种方案：要确保模型更新后的性能至少保持不变或有所提高


                getClusters_UpdateLowA(obj); % 根据目前的GGSOM网络得到一个聚类标签，并使用这些标签更新数据集的下近似信息，同时更新对应节点的下近似信息

                %% 迭代训练GGSOM--对GGSOM进行更新（判断是否生成新的节点）

                epoch=1; % 初始化当前训练轮次
                learn_rate = obj.init_lr * exp(- epoch/obj.Max_iteration); %传统的指数衰减函数
                dynamicThreshold = obj.initial_threshold * (1 - exp(-epoch / obj.Max_iteration)); % 传统的指数衰减函数

                %% 在最大迭代次数内循环
                while epoch<=obj.Max_iteration

                    %% 遍历整个数据集

                    for dataIdx=1:size(obj.testingData, 1)

                        input=obj.testingData(dataIdx,:); % 获取当前的输入向量

                        % 找到最佳匹配单元（BMU）
                        BMU = obj.find_bmu(input); % 输入向量不会变，但在每次迭代中权重向量都会变化，所以BMU会随之变化

                        % 更新获胜神经元的权重
                        BMU.weights = BMU.weights + learn_rate * (input - BMU.weights);

                        %% 获取BMU的邻域神经元并更新其权重
                        findNeighbor2updateWeight(obj, BMU, input, learn_rate, dynamicThreshold);

                        %% 累加误差
                        err = norm(BMU.weights-input);
                        BMU.error = BMU.error + err; % 将给定的误差值累加到节点的误差属性上，并更新误差值

                    end

                    %% 根据误差决定是否进行节点生长

                    for nodeIdx = 1:length(obj.nodes)
                        Node = obj.nodes(1,nodeIdx);
                        if Node.error > obj.GT
                            obj.node_add(Node); % 要确保 obj.Xiajinsi 也同步更新，并且obj.nodes 中节点的顺序和 obj.Xiajinsi 矩阵中行的顺序完全对应。
                            
                            getClusters_UpdateLowA_In_epoch(obj); % 仅在生长节点后更新模糊相似关系和下近似，减少更新频率

                            break; % 将生长节点的频率进行限制，一个迭代只能生长一次
                        end
                    end

                    % getClusters_UpdateLowA_In_epoch(obj); % 在一轮迭代后，根据网络的调整情况重新给测试数据集分配标签，并更新下近似等信息

                    epoch=epoch+1; % 进入下一轮
                    % 更新下一轮的学习率
                    learn_rate = obj.init_lr * exp(- epoch/obj.Max_iteration); %传统的指数衰减函数
                    dynamicThreshold = obj.initial_threshold * (1 - exp(-epoch / obj.Max_iteration)); % 传统的指数衰减函数

                end



                %% 在线特征选择

                % 将聚类结果反映到GGSOM网络的标签信息中
                % clusterResults = getClusterResults(obj, obj.testingData); % 通过GGSOM得到的聚类结果
                % remappedLabels_current = remapLabels(clusterLabels_current, y_current);   % remapLabels函数需要重新被考虑，以确保正确反应聚类结果

                %                 timeND = toc;
                %                 timeND = double(timeND);
                %                 timeND = round(timeND, 4);

                clusterLabelsTTemp = getClusterResults(obj, obj.testingDataCompleted);
                % obj.label = [obj.label; clusterResults]; % 再次更新GGSOM中的label，并将结果传入到GGSOM网络中
                LabelStage = [obj.label; clusterLabelsTTemp];

                % startIdx = size(obj.testingDataCompleted,1)-size(obj.testingData,1)+1;
                % clusterResults = clusterLabelsTTemp(startIdx:end,:);

                %         % 使用混淆矩阵confusion matrix来分析哪些聚类与哪些原始类别最为接近
                %         numClusters = length(unique(clusterResults));
                %         numClusters = max([length(unique(clusterResults)), numClusters]);
                %         numClasses = length(unique(chunkLabels));
                %         confusionMatrix = zeros(numClasses, numClusters);
                %
                %         for i = 1:numClasses
                %             for jj = 1:numClusters
                %                 confusionMatrix(i, jj) = sum(chunkLabels == i & clusterResults == jj);
                %             end
                %         end
                %
                %         % 对于每一个聚类，找到与之最频繁关联的原始类别
                %         [~, bestMatch] = max(confusionMatrix, [], 1);
                %
                %         % 映射clusterResults到新的类别标签
                %         clusterResults(clusterResults > numClusters) = 1;  % 将超界的索引重置为1
                %         mappedLabels = arrayfun(@(x) bestMatch(x), clusterResults);


                % 是否可以改为selectedFeatures = OFS_FNGBRS(obj);
                % 对obj中的下近似、模糊邻域关系矩阵等作最终的更新，并将其信息传入到下一个数据块到来的数据集中。


                % tic;
                [selectedFeatures, obj.Xiajinsi, obj.int_relation]= OFS_FNGBRS111(obj.data, LabelStage);
%                 timeOFS = toc;
%                 timeOFS = double(timeOFS);
%                 timeOFS = round(timeOFS, 4);
                % 对选择出的特征进行分类器上的评价

            end

            setappdata(0, 'GlobalGGSOM', obj);

        end



        function winner=find_bmu(obj, vec)
            %% 获取BMU
            similarity = arrayfun(@(node) fuzzySimilarity(vec, node.weights), obj.nodes);
            [~, winnerIdx] = max(similarity);
            winner = obj.nodes(winnerIdx);
            winner.bmuCount = winner.bmuCount + 1;
        end

        function neighborIndices = findNeighbor2updateWeight(obj, BMU, input, learn_rate, dynamicThreshold)
            %% 获取BMU的邻域神经元并更新其权重

            % 获胜神经元对应的下近似向量
            BMU_Xiajinsi = BMU.LowA;

            % 初始化邻域神经元的索引
            neighborIndices = cell(1, length(obj.nodes)); % 预分配最大可能空间
            neighborCount = 0;

            % 遍历GSOM中的每个神经元
            for i = 1:length(obj.nodes)

                if obj.nodes(1,i).label ~= BMU.label % 跳过BMU本身

                    % 获取当前神经元对应的下近似向量
                    Node_Xiajinsi = obj.nodes(1,i).LowA;

                    %% 计算BMU和当前神经元的下近似隶属度的相似度
                    similarity = calculateSimilarity(BMU_Xiajinsi, Node_Xiajinsi);

                    % 如果相似度高于某个阈值，则认为当前神经元是BMU的邻域神经元
                    if similarity > dynamicThreshold
                        neighborCount = neighborCount + 1;
                        neighborIndices{neighborCount} = obj.nodes(1,i);

                        % 计算影响因子，起到缩放学习率的作用，节点与BMU越相似，权重更新的幅度越大
                        influence = exp(-(1 - similarity)^2 / 2);  % 使用相似度调整影响因子

                        %% 更新邻域神经元的权重
                        obj.nodes(1,i).weights = obj.nodes(1,i).weights + learn_rate * influence * (input - obj.nodes(1,i).weights);
                    end

                end

            end

            neighborIndices = neighborIndices(1:neighborCount); % 调整大小以匹配实际数量

            %% 更新BMU的邻域神经元索引
            BMU.neighbors = neighborIndices;

        end



        function clusterLabels_temp = getClusters_UpdateLowA(obj)
            %% 根据目前的GGSOM网络得到一个聚类标签，并使用这些标签更新数据集的下近似信息，同时更新对应节点的下近似信息
            % clusterLabels_temp = getClusterResults(obj, obj.testingData);
            clusterLabels_temp = getClusterResults(obj, obj.testingDataCompleted);
            temp_label = obj.label; % 临时存储之前的数据块中的标签信息
            Llabel = [temp_label; clusterLabels_temp]; % 更新GGSOM中的label

            %% 判断节点是否活跃
            activeLabels = unique(Llabel); % 筛选出活跃节点，其他节点就是非活跃节点，需要进行删除节点的步骤
            inactiveNodeIndices = zeros(1, length(obj.nodes)); % 预分配内存 用于存储非活跃节点的索引
            inactiveCount = 0; % 用来记录非活跃节点的实际数量
            for i = length(obj.nodes):-1:1
                nodeLabel = obj.nodes(i).label;
                if ~ismember(nodeLabel, activeLabels)
                    inactiveCount = inactiveCount + 1;
                    inactiveNodeIndices(inactiveCount) = i;
                end
            end
            % 调整大小以匹配实际的非活跃节点数量
            inactiveNodeIndices = inactiveNodeIndices(1:inactiveCount);
            obj.nodes(inactiveNodeIndices) = []; % 删除未活跃的节点

            %% 更新下近似
            % myUpdateLowA(obj,Llabel);

            [obj.Xiajinsi,obj.int_relation] = MyNeighbor111(obj.data, Llabel);

            % 更新GGSOM中每个节点的下近似信息
            % 以下循环基于一个重要的假设：即 obj.nodes 中节点的顺序和 obj.Xiajinsi 矩阵中行的顺序完全对应。
            % 要确保每次节点的添加、删除或修改时 obj.Xiajinsi 也同步更新，并且这两者始终按照相同的逻辑维护相同的排序
            for idx = 1:length(obj.nodes)
                % 获取当前节点
                currentNode = obj.nodes(idx);
                % 更新当前节点的下近似信息
                if idx <= size(obj.Xiajinsi, 1)
                    currentNode.LowA = obj.Xiajinsi(idx, :);
                end
            end

        end

        function clusterLabels_temp = getClusters_UpdateLowA_In_epoch(obj)
            %% 根据目前的GGSOM网络得到一个聚类标签，并使用这些标签更新数据集的下近似信息，同时更新对应节点的下近似信息
            clusterLabels_temp = getClusterResults(obj, obj.testingDataCompleted);
            temp_label = obj.label; % 临时存储之前的数据块中的标签信息
            Llabel = [temp_label; clusterLabels_temp]; % 更新GGSOM中的label

            %% 判断节点是否活跃
            activeLabels = unique(Llabel); % 筛选出活跃节点，其他节点就是非活跃节点，需要进行删除节点的步骤
            inactiveNodeIndices = zeros(1, length(obj.nodes)); % 预分配内存 用于存储非活跃节点的索引
            inactiveCount = 0; % 用来记录非活跃节点的实际数量
            for i = length(obj.nodes):-1:1
                nodeLabel = obj.nodes(i).label;
                if ~ismember(nodeLabel, activeLabels)
                    inactiveCount = inactiveCount + 1;
                    inactiveNodeIndices(inactiveCount) = i;
                end
            end
            % 调整大小以匹配实际的非活跃节点数量
            inactiveNodeIndices = inactiveNodeIndices(1:inactiveCount);
            obj.nodes(inactiveNodeIndices) = []; % 删除未活跃的节点

            %% 更新下近似
            % myUpdateLowA_In_epoch(obj,Llabel);
            [obj.Xiajinsi,obj.int_relation] = MyNeighbor111(obj.data, Llabel);

            % 更新GGSOM中每个节点的下近似信息
            % 以下循环基于一个重要的假设：即 obj.nodes 中节点的顺序和 obj.Xiajinsi 矩阵中行的顺序完全对应。
            % 要确保每次节点的添加、删除或修改时 obj.Xiajinsi 也同步更新，并且这两者始终按照相同的逻辑维护相同的排序
            for idx = 1:length(obj.nodes)
                % 获取当前节点
                currentNode = obj.nodes(idx);
                % 更新当前节点的下近似信息
                if idx <= size(obj.Xiajinsi, 1)
                    currentNode.LowA = obj.Xiajinsi(idx, :);
                end
            end
        end

  



        function Add_nodes = node_add(obj, node)
            %% 生长新的节点
            % 注意obj.nodes 中节点的顺序和 obj.Xiajinsi 矩阵中行的顺序完全对应。
            % 要确保每次节点的添加、删除或修改时 obj.Xiajinsi 也同步更新，并且这两者始终按照相同的逻辑维护相同的排序

            % 初始化返回的节点列表，表示可能没有新增节点
            Add_nodes = {}; % 直接在节点周围增长新节点

            %% 计算新节点的标签
            new_label = max([obj.nodes.label]) + 1; % 用已有的标签最大值+1

            %% 计算新节点的初始权重

            % 这种计算方式本质上是在尝试找到一个点，该点在特征空间中位于BMU的权重向量方向上，且远离其邻域节点的平均位置。
            % 这样可以使新节点的权重偏向于当前数据区域的中心，同时又略微偏离其他节点，以保持网络的分布和覆盖性。
            neighborWeight = zeros(1, size(obj.data, 2));
            for neighborIdx = 1:length(node.neighbors)

                neighbo = node.neighbors{neighborIdx};
                neighborWeight = neighborWeight + neighbo.weights;

            end
            initialWeights = 2 * node.weights - neighborWeight / length(node.neighbors);

            %% 计算新节点的初始下近似信息
            % 创建一个全为0的向量，其长度等于数据集的样本数
            LowA = zeros(1, size(obj.data, 1));

            %% 创建新节点并添加到网络中
            new_node = GSOM_Node(new_label, initialWeights, LowA);
            obj.nodes(end+1) = new_node;

            Add_nodes{end+1} = new_node; % 返回的节点列表

            %% 计算新节点的初始误差
            % 分配一部分误差给新增的节点，减少当前节点的误差
            % 在分配给新节点误差以及调整旧节点的误差时，需要考虑到误差应该反映节点在数据结构中的重要性和活跃度。
            % 一个合理的方法是将一部分当前节点的误差传递给新节点，同时确保当前节点的误差减少，以反映它已经部分“释放”了它的误差负担。
            % 这样的分配策略有助于网络逐渐稳定并自我组织。
            % 根据这个思路，我们可以这样设置误差分配逻辑：
            % 新节点的误差：新节点可以继承当前节点的一部分误差，例如当前节点误差的一定比例。这表示新节点在网络中承担了一部分当前节点的表示责任。
            % 当前节点的误差调整：当前节点的误差应当减少，因为它通过生长新节点而“释放”了一部分它的结构压力。

            new_node.error = node.error * obj.gamma; % 正确地设置新节点的误差obj.gamma是误差分布系数
            % 减少当前节点的误差
            node.error = node.error * (1 - obj.gamma);
        end

    end
end


classdef GSOM_Node < handle
    % GSOM_Node类定义了GSOM算法中的单个节点
    properties
        error % 当前节点累积的误差
%         it % 节点被创建时的迭代次数
%         x % 节点的x坐标
%         y % 节点的y坐标
%         last_it % 节点最后一次成为BMU（最佳匹配单元）时的迭代次数
%         last_changed % 最后一次权重调整的迭代次数（此属性在代码中未使用，可能用于后续的拓展）
%         data % 节点存储的数据（此属性在代码中未使用，可能用于后续的拓展）
        weights % 节点的权重向量
        label % 新增属性用于存储节点标签
        neighbors % 存储邻域节点的数组或cell数组
        LowA % 存储节点的下近似信息
        bmuCount % 存储节点的活跃程度-成为BMU的次数
    end

    methods

        function obj=GSOM_Node(nodeLabel, initialWeights, LowApproximate)
            % GSOM_Node的构造函数，初始化一个新节点

            obj.weights = initialWeights; % 初始化权重向量为传入的初始权重

            %Remember the error occuring at this particular node记住在这个特定节点上发生的错误
            obj.error = 0; % 初始化误差为0

            obj.bmuCount = 0;

            obj.LowA = LowApproximate;

%             % Holds the number of the iteration during the node has been inserted.
%             obj.it = 0; % 初始化节点创建时的迭代次数为0
% 
%             %Holds the number of the last iteration where the node has won.
%             obj.last_it = 0; % 初始化节点最后成为BMU的迭代次数为0

%             %Holds the best-matching data.
%             obj.data = -1; % 初始化节点存储的数据（此处-1可能表示未存储任何数据）
%             obj.last_changed = 0; % 初始化最后权重调整的迭代次数为0

            obj.label = nodeLabel; % 保存节点的标签

            obj.neighbors = {};

        end


    end
end
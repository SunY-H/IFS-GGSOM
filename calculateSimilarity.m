function similarity = calculateSimilarity(Xiajinsi_A, Xiajinsi_B)
    % 计算两个下近似集合的相似度
    % Xiajinsi_A 和 Xiajinsi_B 是两个下近似集合的向量

    % 计算交集大小
    intersectSize = sum(min(Xiajinsi_A, Xiajinsi_B));
    
    % 计算并集大小
    unionSize = sum(max(Xiajinsi_A, Xiajinsi_B));
    
    % 计算相似度
    if unionSize == 0
        similarity = 0;  % 避免除以零的情况
    else
        similarity = intersectSize / unionSize;
    end
end

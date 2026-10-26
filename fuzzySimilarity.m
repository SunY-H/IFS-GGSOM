function similarity = fuzzySimilarity(vec1, vec2)
    % 计算两个向量之间的欧式距离
    distance = norm(vec1 - vec2);
    
    % 将距离转换为相似度，使用距离的倒数作为相似度度量
    % 注意避免除以零的情况
    if distance == 0
        similarity = Inf; % 完全相同的向量相似度为无穷大
    else
        similarity = 1 / distance;
    end
end


% function similarity = fuzzySimilarity(vec1, vec2)
% % 使用模糊相似度来度量输入向量与权重向量之间的相似性。
% minSum = 0;
% maxSum = 0;
% for i = 1:length(vec1)
%     minSum = minSum + min(vec1(i), vec2(i));
%     maxSum = maxSum + max(vec1(i), vec2(i));
% end
% similarity = minSum / maxSum;  % 计算加权相似度
% end

%         function similarity = weightedFuzzySimilarity(vec1, vec2, weights)
%             % 使用加权模糊相似度来度量输入向量与权重向量之间的相似性。这里的权重可以是下近似度，它反映了每个特征对于类别的重要性。
%             minSum = 0;
%             maxSum = 0;
%             for i = 1:length(vec1)
%                 minSum = minSum + weights(i) * min(vec1(i), vec2(i));
%                 maxSum = maxSum + weights(i) * max(vec1(i), vec2(i));
%             end
%             similarity = minSum / maxSum;  % 计算加权相似度
%         end



%         function similarity = roughSetSimilarity(inputVec, classCenter, XiajinsiRow)
%             % 利用模糊粗糙集理论，可以考虑输入向量与各个类别下近似或上近似集合的关系来度量相似性。
%             % 计算输入向量与类中心之间的相似度，考虑下近似度
%             similarity = dot(inputVec, classCenter) / (norm(inputVec) * norm(classCenter));  % 例如使用余弦相似度
%             similarity = similarity * mean(XiajinsiRow);  % 考虑下近似度的影响
%         end

% function similarity = fuzzySimilarity(vec1, vec2)
%     
%     % 用余弦相似度作为衡量两个向量之间相似度的方法
% 
%     % 确保输入向量是行向量或列向量
%     if isrow(vec1)
%         vec1 = vec1';
%     end
%     if isrow(vec2)
%         vec2 = vec2';
%     end
%     
%     % 计算两个向量的点积
%     dotProduct = dot(vec1, vec2);
%     
%     % 计算两个向量的模
%     normVec1 = norm(vec1);
%     normVec2 = norm(vec2);
%     
%     % 计算余弦相似度
%     similarity = dotProduct / (normVec1 * normVec2);
% end

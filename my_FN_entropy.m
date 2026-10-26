function FNE = my_FN_entropy(row,fuzzy_neighbor_granule,FD,class_num)

%% Ä£ºýÁÚÓòìØ

FNE = 0;

for k = 1:row
    fuzzy_neighbor_temp1 = fuzzy_neighbor_granule(k,:);
    FN_granual1 = find(fuzzy_neighbor_temp1 ~= 0);
    FN_zeronum1 = find(fuzzy_neighbor_temp1 == 0);
    for j = 1:class_num
        FD_temp = FD(:,j)';
        number = find((fuzzy_neighbor_temp1 - FD_temp) <= 0);
        n2 = length(number) - length(FN_zeronum1) ;
        
        if n2 > 0 && n2 <= length(FN_granual1)
            n1 = 1 / row;
             Joint_en_temp = -n1 * log2(n1 * n2);
             FNE = FNE + Joint_en_temp;
        end
    end
end

end


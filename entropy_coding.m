fid1 = fopen('sequence_example.txt','w');
fid2 = fopen('encoded_code_example.txt','w');
EOF = 12; 
size = 5;
nbtest = 40;

probs = [5, 50, 40, 5, 5, 50, 40, 5, 5, 50, 40, 5];
bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, EOF];

for idx=1:nbtest
    seq = randi(11, 1, size);
    seq = [seq EOF];
    code = arithenco(seq,probs);
    dseq = arithdeco(code,probs,length(seq));
    isequal(seq, dseq) 
    fprintf(fid1,'%s\n',num2str(seq));
    fprintf(fid2,'%s\n',num2str(code));
end
fclose(fid1);
fclose(fid2);
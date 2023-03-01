function [Y_noisy] = tvg_realdata(path_directory,N,num)
% N maxNumPoints
    snr = 10;
    original_files=dir([path_directory '/*.off']); 
    filename=[path_directory '/' original_files(1).name];
    Y_noisy =ptcloud_process(filename,N,snr);
    for k=2:min(num,length(original_files))
        filename=[path_directory '/' original_files(k).name];
        y_noisy = ptcloud_process(filename,N,snr);
        Y_noisy = cat(2,Y_noisy,y_noisy);
    end
end

function [y_noisy]= ptcloud_process(filename,N,snr)
        [vertex,face] = read_off(filename);
        v = vertex';
        ptCloudIn = pointCloud(v);
        ptCloudOut = pcdownsample(ptCloudIn,'nonuniformGridSample',N);
        y = ptCloudOut.Location(:,2);
        y_noisy = awgn(y,snr);
end

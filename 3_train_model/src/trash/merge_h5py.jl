using HDF5

inputfilepath=ARGS[1]
outputfilepath=ARGS[2]
# outputfilepath=ARGS[2]

print(inputfilepath)
h5open(outputfilepath, "w") do out_file:
  f = open(inputfilepath)
  firstit=true
  data=[]
  for line in eachline(f)
      r = strip(line, ['\n'])
      print(r,"\n")
      in_file = h5open("/mnt/brain4/datasets/Twitter/final/text/838713814430404608_838778372993929216.h5", "r")
      for name in names(in_file)
        write(out_file, name, )
        # datai = h5read(r, "/data")
#       if (firstit)
#           data=datai
#           firstit=false
#       else
#           data=cat(4,data, datai) #In this case concatenating on 4th dimension
#       end
  end

# end
# h5write(outputfilepath, "/data", data)
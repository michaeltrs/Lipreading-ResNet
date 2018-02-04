local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'

local function createModel(opt)
   local model = nn.Sequential()
   --[[ 
   cudnn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH)
   https://nn.readthedocs.io/en/rtd/convolution/index.html#nn.VolumetricConvolution
    Applies a 3D convolution over an input image composed of several input planes. The input tensor in forward(input) is expected to be a 4D tensor (nInputPlane x time x height x width).

    The parameters are the following: 
        * nInputPlane: The number of expected input planes in the image given into forward(). 1
        * nOutputPlane: The number of output planes the convolution layer will produce. 64
        * kT: The kernel size of the convolution in time 5
        * kW: The kernel width of the convolution 7
        * kH: The kernel height of the convolution 7
        * dT: The step of the convolution in the time dimension. Default is 1. 1
        * dW: The step of the convolution in the width dimension. Default is 1. 2
        * dH: The step of the convolution in the height dimension. Default is 1.2

    Note that depending of the size of your kernel, several (of the last) columns or rows of the input image might be lost. It is up to the user to add proper padding in images.

    If the input image is a 4D tensor nInputPlane x time x height x width, the output image size will be nOutputPlane x otime x owidth x oheight where
   ]]--
   model:add(cudnn.VolumetricConvolution(1, 64, 5, 7, 7, 1, 2, 2, 2, 3, 3))


   --[[
   cudnn.VolumetricBatchNormalization(nFeature, eps, momentum, affine)

   ]]--
   model:add(cudnn.VolumetricBatchNormalization(64))
   model:add(nn.ReLU(true))
   
   --[[
   nn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH [, dT, dW, dH])

   ]]--
   model:add(cudnn.VolumetricMaxPooling(1,3,3,1,2,2,0,1,1))

   local function ConvInitV(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kT*v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   ConvInitV('cudnn.VolumetricConvolution')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()
   return model
end

return createModel

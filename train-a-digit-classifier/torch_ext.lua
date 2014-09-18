
require 'nn'
require 'nnx'

function nn.Sequential:getOutput(input)
   local output_seq = {}
   local currentOutput = input
   for i=1,#self.modules do
      currentOutput = self.modules[i]:updateOutput(currentOutput)
      output_seq[i] = currentOutput:clone()
   end
   return output_seq
end

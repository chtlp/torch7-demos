----------------------------------------------------------------------
-- simple utility library for lua
require 'pl'

-- clear all key/value pairs from a table, not just the 1-based array
function tablex.clearall(t)
   for k in pairs(t) do
      t[k] = nil
   end
end

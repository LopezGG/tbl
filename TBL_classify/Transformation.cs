using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TBL_train
{
    class Transformation
    {
        public String TriggerFeature;
        public String FromClass;
        public String ToClass;
        public int netGain;

        public Transformation (string TF, string FC,string TC,int sc)
        {
            TriggerFeature = TF;
            FromClass = FC;
            ToClass = TC;
            netGain = sc;
        }

    }
}

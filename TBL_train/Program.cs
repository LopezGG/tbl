using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TBL_train
{
    class Program
    {
        static void Main (string[] args)
        {
            string TrainData = args[0];
            string ModelFile = args[1];
            double MinGain = Convert.ToDouble(args[2]);
            Stopwatch clock = new Stopwatch();
            clock.Start();
            int docId = 0;
            Dictionary<int,Document> TrainingDocs = new Dictionary<int,Document>();
            Dictionary<string, List<int>> classDocId = new Dictionary<string,List<int>>();
            List<String> Vocab = new List<String>();
            List<String> UniqueClass = new List<String>();


            //read training Docs and create Vocab , list of trainingDocs and list of unique classes
            ReadTraining (ref TrainingDocs, ref Vocab,ref UniqueClass,TrainData, ref docId,ref classDocId);
            //training starts here
            List<Transformation> TransformationList = new List<Transformation>();
            Console.WriteLine("timeElapsed: " + clock.Elapsed);
            while(true)
            {
                Transformation curTransformation = GetBestTransformation ( Vocab, UniqueClass, TrainingDocs,classDocId);
                Console.WriteLine("timeElapsed: " + clock.Elapsed+" gain: " + curTransformation.netGain);
                if(curTransformation==null)
                {
                    Console.WriteLine("no Candidates");
                    break;
                }
                if ((curTransformation.netGain) >= MinGain)
                {
                    
                    List<int> DocIdToUpdate;
                    if (classDocId.ContainsKey(curTransformation.FromClass))
                        DocIdToUpdate = classDocId[curTransformation.FromClass].ToList();
                    else
                        break;
                    //we will go ahead and do this transformation so we update the existing correct instances score, add to list of transformations and change the curClass of doc.
                    TransformationList.Add(curTransformation);
                    foreach (var dId in DocIdToUpdate)
                    {
                        if (TrainingDocs[dId].FeatureDict.ContainsKey(curTransformation.TriggerFeature) && TrainingDocs[dId].curClass == curTransformation.FromClass)
                        {
                            TrainingDocs[dId].curClass = curTransformation.ToClass;
                            classDocId[curTransformation.FromClass].Remove(dId);
                            classDocId[curTransformation.ToClass].Add(dId);
                        }
                        
                    }
                }
                //if we can find no transformation with higher than expected gain, just let us stop.
                else
                    break;
            }
            int correctClass = 0;
            foreach (var doc in TrainingDocs)
            {
                if(doc.Value.curClass == doc.Value.correctClass)
                    correctClass++;
            }
            using(StreamWriter Sw = new StreamWriter(ModelFile))
            {
                foreach (var trans in TransformationList)
                {
                    Sw.WriteLine(trans.TriggerFeature + " " + trans.FromClass + " " + trans.ToClass + " " + trans.netGain);
                }
            }
    


            clock.Stop();
            Console.WriteLine("Accuracy : " + Convert.ToString(correctClass/((double)docId)));
            Console.WriteLine("Time Elapsed: {0} ", clock.Elapsed);
            Console.ReadLine();

        }

        public static Transformation GetBestTransformation (List<String> Vocab, List<String> UniqueClass, Dictionary<int, Document> TrainingDocs, Dictionary<string, List<int>> classDocId)
        {
            
            if (Vocab.Count == 0 || UniqueClass.Count == 0 || TrainingDocs.Count == 0)
                return null;
            List<Transformation> TL = new List<Transformation>();
            int score;
            foreach (var feature in Vocab)
            {
                foreach (var fromClass in UniqueClass)
                {
                    List<int> requiredDocs;
                    if (classDocId.ContainsKey(fromClass) && classDocId[fromClass].Count > 0)
                        requiredDocs = classDocId[fromClass];
                    else
                        continue;
                    foreach (var toClass in UniqueClass)
                    {
                        //this basically means  no change
                        if (fromClass == toClass)
                            continue;
                        score = 0;
                        foreach (var docid in requiredDocs)
                        {
                            var doc = TrainingDocs[docid];
                            //we will do the transformation only if it contains the feature and the class
                            if (doc.FeatureDict.ContainsKey(feature) && doc.curClass == fromClass)
                            {
                                //if curClass = toClass, I will either relabel it correctly or relabel it incorrectly. my netgain is zero
                                if (doc.curClass == toClass)
                                    continue;
                                //if this change is going to change it to correct class my score increases
                                if (doc.correctClass == toClass)
                                    score++;
                                //I already have the correctClass which is not equal to toClass. So changing to toClass will reduce my score
                                else if (doc.curClass == doc.correctClass)
                                    score--;
                            }
                            //else I do nothing. this means I willneither gain nor loose 
                        }
                        if(score>0)
                            TL.Add(new Transformation(feature, fromClass, toClass, score));
                    }
                }
            }
            if (TL.Count > 0)
                return (TL.OrderByDescending(x => x.netGain).First());
            else
                return null;
        }

        public static void ReadTraining (ref Dictionary<int, Document> TrainingDocs, ref List<String> Vocab,ref List<String> UniqueClass, string TrainData, ref int docId,ref Dictionary<string, List<int>> classDocId)
        {
            string line, word,firstClass="";
            bool flag = true;
            using (StreamReader Sr = new StreamReader(TrainData))
            {
                while ((line = Sr.ReadLine()) != null)
                {
                    if (String.IsNullOrWhiteSpace(line))
                        continue;
                    string[] words = line.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries);
                    if (flag)
                    {
                        firstClass = words[0];
                        classDocId.Add(firstClass, new List<int>());
                        flag = false;
                    }
                    Document temp = new Document(words[0], firstClass);
                    UniqueClass.Add(words[0]);
                    for (int i = 1; i < words.Length; i++)
                    {
                        word = words[i].Substring(0, words[i].IndexOf(":"));
                        Vocab.Add(word);
                        if (!temp.FeatureDict.ContainsKey(word))
                            temp.FeatureDict.Add(word, true);
                    }
                    classDocId[firstClass].Add(docId);
                    TrainingDocs.Add(docId,temp);
                    docId++;
                }
                Vocab = Vocab.Distinct().ToList();
                UniqueClass = UniqueClass.Distinct().ToList();

                foreach (var cl in UniqueClass)
                {
                    if (!classDocId.ContainsKey(cl))
                        classDocId.Add(cl, new List<int>());
                }
            }
        }
    }
}

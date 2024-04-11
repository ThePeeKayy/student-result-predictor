'use client'
import { motion } from 'framer-motion'
import Image from "next/image";
import SelectMenu from "./ui/SelectMenu";
import { slideIn,staggerContainer,textVariant2,textContainer } from '@/utils/motion';
import { FaGithub,FaArrowRight } from "react-icons/fa6";

const LunchOption = [
  { id: 1, name: 'Premium' },
  { id: 2, name: 'Free/Reduced' },

]
const EducationOption = [
  { id: 1, name: 'Wade Cooper' },
  { id: 2, name: 'Arlene Mccoy' },
  { id: 3, name: 'Devon Webb' },
  { id: 4, name: 'Tom Cook' },
  { id: 5, name: 'Tanya Fox' },
]

const RaceOption = [
  { id: 1, name: 'Group A' },
  { id: 2, name: 'Group B' },
  { id: 3, name: 'Group C' },
  { id: 4, name: 'Group D' },

]


export default function Home() {
  return (
    <main 
    className="bg-gray-900 flex overflow-hidden h-[100vh] justify-center"
    >
      <motion.div
      variants={staggerContainer}
      initial='hidden'
      whileInView='show'
      viewport={{once:false,amount:0.25}}
      className={`bg-gray-800 bg-opacity-95 rounded-lg p-4 max-w-[1000px] shadow-lg my-8 w-[92vw] flex space-y-4 flex-col`}
      >
        <div className='flex flex-row space-x-4 ml-[70px]'>
        <Image width={90} height={50} src='/Our-story.png'/>
      <motion.p 
        variants={textContainer}
        className={`font-bold flex flex-row flex-wrap text-transparent bg-gradient-to-r from-red-400 to-yellow-300 bg-clip-text text-[50px] `}
      >
        {Array.from('Student Result Predictor').map((letter,i) => (
          <motion.span variants ={textVariant2} className='flex flex-wrap whitespace-pre-wrap' key={i}>
            {letter === ' '?'\u00A0' :letter}
          </motion.span>
        ))}
        
      </motion.p>
      </div>
        <p className='text-white px-16 text-lg text-justify font-sans'>Explore the future of education with our innovative math score prediction bot. Using cutting-edge technology, our bot analyzes reading and writing scores along with parental education levels to make precise predictions, helping students and educators alike excel.</p>
      <form className='gap-2 flex-col flex px-16'>
      <input
          type="number"
          name="reading score"
          id="reading score"
          className="pl-2 mt-3 block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
          placeholder="Reading Score"
        />
        <input
          type="number"
          name="writing score"
          id="writing score"
          className="pl-2 block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
          placeholder="Writing score"
        />
        <SelectMenu title={'Education level'} options={EducationOption}/>
        <SelectMenu title={'Race Group'} options={RaceOption}/>
        <SelectMenu title={'Lunch Option'} options={LunchOption}/>
      </form>
      <div className='flex justify-center'>
        <div className='bg-white font-bold font-sans rounded-md w-[30vw] min-w-[100px] p-2 ring-[2px] ring-gray-500 shadow-xl'>
          Expected Math Result:
        </div>
      </div>
      <div className='flex flex-col items-center'>
        <span className='flex flex-row text-gray-400'>Chun Yin <FaGithub size={20}/> </span>
        <span className='flex flex-row text-gray-400'>PengKiang <FaGithub size={20}/></span>
        <a className='bg-white p-2 rounded-md mt-2 flex flex-row gap-x-2 font-[600px]' href='https://www.kaggle.com/datasets/spscientist/students-performance-in-exams'>Visit Dataset <div className='bg-gray-800 rounded-full justify-center items-center pt-1 px-1'><FaArrowRight color='white'/></div></a>
      </div>
      <motion.div
      variants={slideIn('right','tween',0.2,1)}
      className="relative flex justify-end ml-[90px]"
      >
        <Image className='absolute right-[-10px] bottom-[-30px]' width={150} height={150} src='/robo.png'/>
      </motion.div>
      
      </motion.div>
    </main>
    
  );
}

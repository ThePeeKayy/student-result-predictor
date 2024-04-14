'use client'
import { motion } from 'framer-motion'
import Image from "next/image";
import SelectMenu from "./ui/SelectMenu";
import { slideIn,staggerContainer,textVariant2,textContainer } from '@/utils/motion';
import { FaGithub,FaArrowRight } from "react-icons/fa6";
import { useState } from 'react';

const LunchOption = [
  { id: 1, name: 'standard' },
  { id: 2, name: 'free/reduced' },

]

const TestOption = [
  { id: 1, name: 'completed' },
  { id: 2, name: 'none' },

]

const GenderOption = [
  { id: 1, name: 'male' },
  { id: 2, name: 'female' },

]
const EducationOption = [
  { id: 1, name: 'high school' },
  { id: 2, name: 'bachelor\'s degree' },
  { id: 3, name: 'master\'s degree' },
  { id: 4, name: 'some high school' },
  { id: 5, name: 'associate\'s degree' },
  { id: 6, name: 'some college' },
]

const RaceOption = [
  { id: 1, name: 'group A' },
  { id: 2, name: 'group B' },
  { id: 3, name: 'group C' },
  { id: 4, name: 'group D' },
  { id: 5, name: 'group E' },
]


export default function Home() {
    const [readingResult,setReadingResult] = useState(null)
    const [writingResult,setWritingResult] = useState(null)
    const [testOption,setTestOption] = useState('completed')
    const [raceOption,setRaceOption] = useState('group A')
    const [lunchOption,setLunchOption] = useState('standard')
    const [genderOption,setGenderOption] = useState('male')
    const [educationOption,setEducationOption] = useState('high school')
    const [mathResult,setMathResult] = useState(null)
    const handleSubmit = async () => {
      const response = await fetch('https://peekay123.pythonanywhere.com/', {
        method:'POST',
        headers: {
          'Content-type':'application/json'
        },
        body:JSON.stringify({
          readingResult:readingResult,
          writingResult:writingResult,
          testOption:testOption,
          raceOption:raceOption,
          lunchOption:lunchOption,
          genderOption:genderOption,
          educationOption:educationOption,
        })
  
      })

      if (response.statusCode== 500) return;
  
      const data = await response.json();
      setMathResult(data.mathResult)
    }
  
  return (
    <main 
    className="bg-gray-900 flex overflow-auto lg:overflow-hidden lg:h-[100vh] justify-center"
    >
      <motion.div
      variants={staggerContainer}
      initial='hidden'
      whileInView='show'
      viewport={{once:false,amount:0.25}}
      className={`bg-gray-800 bg-opacity-95 rounded-lg p-4 max-w-[1000px] shadow-lg my-8 w-[92vw] flex space-y-4 flex-col`}
      >
        <div className='flex flex-row space-x-4 ml-[30px] lg:ml-[70px]'>
        <Image className='hidden lg:block' width={90} height={50} src='/Our-story.png'/>
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
          onChange={(e)=>setReadingResult(e.target.value)}
          required
        />
        <input
          type="number"
          name="writing score"
          id="writing score"
          className="pl-2 block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
          placeholder="Writing score"
          onChange={(e)=>setWritingResult(e.target.value)}
          required
        />
        <div className='lg:flex-row lg:gap-x-3 flex-col flex'>
          <div className='flex flex-col lg:w-[50%] w-full space-y-2'>
            <SelectMenu setEducationOption={setEducationOption} title={'Education level'} options={EducationOption}/>
            <SelectMenu setGenderOption={setGenderOption} title={'Gender'} options={GenderOption}/>
            <SelectMenu setTestOption={setTestOption} title={'Test Prep'} options={TestOption}/>
          </div>
          <div className='flex flex-col lg:w-[50%] w-full space-y-2'> 
            <SelectMenu setRaceOption={setRaceOption} title={'Race Group'} options={RaceOption}/>
            <SelectMenu setLunchOption={setLunchOption} title={'Lunch Option'} options={LunchOption}/>
            <div className='pt-8'>
            <div onClick={()=>handleSubmit()} className='bg-black flex flex-row justify-between ring-inset ring-1 text-sm px-3 py-2 ring-gray-400 text-white rounded-md cursor-pointer'>Submit <div className='bg-white rounded-full justify-center items-center pt-[3px] px-1'><FaArrowRight color='black'/></div></div>
            </div>
          </div>
        </div>
      </form>
      <div className='flex justify-center'>
        <div className='bg-white font-bold font-sans rounded-md w-[30vw] min-w-[100px] p-2 ring-[2px] ring-gray-500 shadow-xl'>
          Expected Math Result: {mathResult ? mathResult.toFixed(1): null}
        </div>
      </div>
      <div className='flex flex-col items-center'>
        <a href='https://github.com/ThePeeKayy' className='flex flex-row text-gray-400'>PengKiang <FaGithub size={20}/></a>

        <a href='https://github.com/iextremity' className='flex flex-row text-gray-400'>Chun Yin <FaGithub size={20}/> </a>
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

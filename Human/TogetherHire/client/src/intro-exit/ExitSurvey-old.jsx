import { usePlayer } from "@empirica/core/player/classic/react";
import React, { useState } from "react";
import { Alert } from "../components/Alert.jsx";
import { Button } from "../components/Button.jsx";

export function ExitSurvey({ next }) {
  const labelClassName = "block text-sm font-medium text-gray-700 my-2";
  const inputClassName =
    "appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-empirica-500 focus:border-empirica-500 sm:text-sm";
  const player = usePlayer();

  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [education, setEducation] = useState("");
  const [country, setCountry] = useState("");
  const [experience, setExperience] = useState("");
  const [motivation, setMotivation] = useState("");
  const [engagement, setEngagement] = useState("");
  const [easeOfUse, setEaseOfUse] = useState("");
  const [difficulty, setDifficulty] = useState("");
  const [interference, setInterference] = useState("");
  const [strategy, setStrategy] = useState("");
  const [keyFactors, setKeyFactors] = useState("");
  const [comfort, setComfort] = useState("");
  const [clarity, setClarity] = useState("");
  const [timeFeedback, setTimeFeedback] = useState("");
  const [dataConsent, setDataConsent] = useState("");
  const [followUp, setFollowUp] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    player.set("exitSurvey", {
      age,
      gender,
      education,
      country,
      experience,
      motivation,
      engagement,
      easeOfUse,
      difficulty,
      interference,
      strategy,
      keyFactors,
      comfort,
      clarity,
      timeFeedback,
      dataConsent,
      followUp,
    });
    next();
  }

  function handleEducationChange(e) {
    setEducation(e.target.value);
  }

  return (
    <div className="py-8 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
      <Alert title="Bonus">
        <p>
          Please submit the following code to receive your bonus:{" "}
          <strong>{player.id}</strong>.
        </p>
        <p className="pt-1">
          Your final <strong>bonus</strong> is in addition to the{" "}
          <strong>1 base reward</strong> for completing the HIT.
        </p>
      </Alert>

      <form
        className="mt-12 space-y-8 divide-y divide-gray-200"
        onSubmit={handleSubmit}
      >
        <div className="space-y-8 divide-y divide-gray-200">
          <div>
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Exit Survey
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              Please answer the following short survey. You do not have to
              provide any information you feel uncomfortable with.
            </p>

            <div className="space-y-8 mt-6">
              {/* Demographics */}
              <div className="flex flex-row">
                <div>
                  <label htmlFor="age" className={labelClassName}>
                    Age
                  </label>
                  <input
                    id="age"
                    name="age"
                    type="number"
                    autoComplete="off"
                    className={inputClassName}
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                  />
                </div>
                <div className="ml-5">
                  <label htmlFor="gender" className={labelClassName}>
                    Gender
                  </label>
                  <input
                    id="gender"
                    name="gender"
                    autoComplete="off"
                    className={inputClassName}
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                  />
                </div>
              </div>

              <div>
                <label className={labelClassName}>
                  Highest Education Qualification
                </label>
                <div className="grid gap-2">
                  <Radio
                    selected={education}
                    name="education"
                    value="high-school"
                    label="High School"
                    onChange={handleEducationChange}
                  />
                  <Radio
                    selected={education}
                    name="education"
                    value="bachelor"
                    label="Bachelor's Degree"
                    onChange={handleEducationChange}
                  />
                  <Radio
                    selected={education}
                    name="education"
                    value="master"
                    label="Master's or higher"
                    onChange={handleEducationChange}
                  />
                  <Radio
                    selected={education}
                    name="education"
                    value="other"
                    label="Other"
                    onChange={handleEducationChange}
                  />
                </div>
              </div>

              <div>
                <label htmlFor="country" className={labelClassName}>
                  Primary Country/Region of Residence
                </label>
                <input
                  id="country"
                  name="country"
                  autoComplete="off"
                  className={inputClassName}
                  value={country}
                  onChange={(e) => setCountry(e.target.value)}
                />
              </div>

              {/* Experiment-Related */}
              <div>
                <label className={labelClassName}>
                  Have you participated in similar experiments before?
                </label>
                <div className="grid gap-2">
                  <Radio
                    selected={experience}
                    name="experience"
                    value="yes"
                    label="Yes"
                    onChange={(e) => setExperience(e.target.value)}
                  />
                  <Radio
                    selected={experience}
                    name="experience"
                    value="no"
                    label="No"
                    onChange={(e) => setExperience(e.target.value)}
                  />
                </div>
              </div>

              <div>
                <label htmlFor="motivation" className={labelClassName}>
                  What motivated you to participate in this experiment?
                </label>
                <input
                  id="motivation"
                  name="motivation"
                  autoComplete="off"
                  className={inputClassName}
                  value={motivation}
                  onChange={(e) => setMotivation(e.target.value)}
                />
              </div>

              {/* Game Experience */}
              <div>
                <label className={labelClassName}>
                  How engaging was the game? (1 = Not engaging, 5 = Very
                  engaging)
                </label>
                <input
                  type="number"
                  min="1"
                  max="5"
                  className={inputClassName}
                  value={engagement}
                  onChange={(e) => setEngagement(e.target.value)}
                />
              </div>

              <div>
                <label className={labelClassName}>
                  How easy was it to use the game interface? (1 = Very
                  difficult, 5 = Very easy)
                </label>
                <input
                  type="number"
                  min="1"
                  max="5"
                  className={inputClassName}
                  value={easeOfUse}
                  onChange={(e) => setEaseOfUse(e.target.value)}
                />
              </div>

              <div>
                <label className={labelClassName}>
                  How difficult was the task? (1 = Very easy, 5 = Very
                  difficult)
                </label>
                <input
                  type="number"
                  min="1"
                  max="5"
                  className={inputClassName}
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                />
              </div>

              <div>
                <label htmlFor="interference" className={labelClassName}>
                  Did you encounter any technical issues or distractions? If
                  yes, please describe.
                </label>
                <textarea
                  id="interference"
                  name="interference"
                  rows="4"
                  className={inputClassName}
                  value={interference}
                  onChange={(e) => setInterference(e.target.value)}
                />
              </div>

              {/* Decision-Making */}
              <div>
                <label htmlFor="strategy" className={labelClassName}>
                  How did you approach decision-making in this experiment?
                </label>
                <textarea
                  id="strategy"
                  name="strategy"
                  rows="4"
                  className={inputClassName}
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                />
              </div>

              <div>
                <label htmlFor="keyFactors" className={labelClassName}>
                  What key factors influenced your decisions?
                </label>
                <textarea
                  id="keyFactors"
                  name="keyFactors"
                  rows="4"
                  className={inputClassName}
                  value={keyFactors}
                  onChange={(e) => setKeyFactors(e.target.value)}
                />
              </div>

              {/* Environment Feedback */}
              <div>
                <label className={labelClassName}>
                  How comfortable was the experiment environment? (1 = Not
                  comfortable, 5 = Very comfortable)
                </label>
                <input
                  type="number"
                  min="1"
                  max="5"
                  className={inputClassName}
                  value={comfort}
                  onChange={(e) => setComfort(e.target.value)}
                />
              </div>

              <div>
                <label className={labelClassName}>
                  How clear were the instructions? (1 = Very unclear, 5 = Very
                  clear)
                </label>
                <input
                  type="number"
                  min="1"
                  max="5"
                  className={inputClassName}
                  value={clarity}
                  onChange={(e) => setClarity(e.target.value)}
                />
              </div>

              <div>
                <label className={labelClassName}>
                  Was the time allocated for the experiment reasonable? (1 =
                  Very unreasonable, 5 = Very reasonable)
                </label>
                <input
                  type="number"
                  min="1"
                  max="5"
                  className={inputClassName}
                  value={timeFeedback}
                  onChange={(e) => setTimeFeedback(e.target.value)}
                />
              </div>

              {/* Ethics */}
              <div>
                <label className={labelClassName}>
                  Do you consent to the use of your data for research purposes?
                </label>
                <div className="grid gap-2">
                  <Radio
                    selected={dataConsent}
                    name="dataConsent"
                    value="yes"
                    label="Yes"
                    onChange={(e) => setDataConsent(e.target.value)}
                  />
                  <Radio
                    selected={dataConsent}
                    name="dataConsent"
                    value="no"
                    label="No"
                    onChange={(e) => setDataConsent(e.target.value)}
                  />
                </div>
              </div>

              <div>
                <label className={labelClassName}>
                  Would you like to participate in follow-up studies?
                </label>
                <div className="grid gap-2">
                  <Radio
                    selected={followUp}
                    name="followUp"
                    value="yes"
                    label="Yes"
                    onChange={(e) => setFollowUp(e.target.value)}
                  />
                  <Radio
                    selected={followUp}
                    name="followUp"
                    value="no"
                    label="No"
                    onChange={(e) => setFollowUp(e.target.value)}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mb-12">
          <Button type="submit">Submit</Button>
        </div>
      </form>
    </div>
  );
}

export function Radio({ selected, name, value, label, onChange }) {
  return (
    <label className="text-sm font-medium text-gray-700">
      <input
        className="mr-2 shadow-sm sm:text-sm"
        type="radio"
        name={name}
        value={value}
        checked={selected === value}
        onChange={onChange}
      />
      {label}
    </label>
  );
}
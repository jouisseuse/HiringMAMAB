// src/exit/ExitSurvey.jsx
import { usePlayer } from "@empirica/core/player/classic/react";
import React, { useState } from "react";
import { Button } from "../components/Button";
import { Alert } from "../components/Alert.jsx";

const REDIRECT_URL = "https://connect.cloudresearch.com/participant/project/57A8FE2C72/complete"; // ← change me
const OPEN_IN_NEW_TAB = false; // set true to open in a new tab

export function ExitSurvey({ next }) {
  const labelClassName = "block text-sm font-medium text-gray-700 my-2";
  const inputClassName =
    "appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-empirica-500 focus:border-empirica-500 sm:text-sm";
  const player = usePlayer();

  const UNIVERSITY_OPTIONS = [
    "Trinity College Dublin",
    "The University of Western Australia",
    "University of Glasgow",
    "Heidelberg University",
    "University of Adelaide",
    "University of Leeds",
    "University of Southampton",
    "University of Sheffield",
    "University of Nottingham",
    "Karlsruhe Institute of Technology",
    "None of the above",
  ];

  const EMPLOYMENT_OPTIONS = [
    "Student",
    "Employed full-time",
    "Employed part-time",
    "Self-employed",
    "Unemployed",
    "Other",
  ];

  const HIRING_YEARS = [
    "Less than 1 year",
    "1–3 years",
    "4–6 years",
    "7+ years",
  ];

  const [currentPage, setCurrentPage] = useState(1);
  const [formData, setFormData] = useState({
    // Page 1
    age: "",
    gender: "",
    genderOther: "",
    race: "",
    raceOther: "",
    education: "",
    educationOther: "",
    country: "",
    politicalOrientation: "",
    // (original Page 2 content, now Page 3)
    experience: "",
    engagement: "",
    clearity: "",
    interference: "",
    interferenceOption: "",
    keyFactors: "",
    strategy: "",
    motivation: "",
    followUp: "",
    additionalFeedback: "",
    experienceDetails: "",
    // (original Page 3 content, now Page 2)
    familiarityUniversities: [],
    employmentStatus: "",
    employmentStatusOther: "",
    hiringInvolved: "",
    hiringExperienceYears: "",
  });

  const [errors, setErrors] = useState({});

  const handleInputChange = (key, value) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
    setErrors((prev) => ({ ...prev, [key]: "" }));
  };

  // Why: "None of the above" must be mutually exclusive.
  const toggleUniversity = (name) => {
    setFormData((prev) => {
      const current = new Set(prev.familiarityUniversities);
      if (current.has(name)) {
        current.delete(name);
      } else {
        if (name === "None of the above") {
          current.clear();
          current.add(name);
        } else {
          current.delete("None of the above");
          current.add(name);
        }
      }
      return { ...prev, familiarityUniversities: Array.from(current) };
    });
    setErrors((prev) => ({ ...prev, familiarityUniversities: "" }));
  };

  const handleNextPage = () => {
    const validationErrors = validatePage(currentPage);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }
    setCurrentPage((prev) => prev + 1);
  };

  const handlePreviousPage = () => {
    setCurrentPage((prev) => prev - 1);
  };

  // const handleSubmit = (event) => {
  //   event.preventDefault();
  //   const validationErrors = validatePage(currentPage);
  //   if (Object.keys(validationErrors).length > 0) {
  //     setErrors(validationErrors);
  //     return;
  //   }
  //   player.set("exitSurvey", formData);
  //   next();
  // };
  // Replace your current handleSubmit with this:
  const handleSubmit = (event) => {
    event.preventDefault();

    const validationErrors = validatePage(currentPage);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    // Persist survey to Empirica
    player.set("exitSurvey", formData);

    // Advance Empirica flow (if needed)
    next?.();

    // Redirect after a short tick so state/WS flush can happen
    setTimeout(() => {
      if (OPEN_IN_NEW_TAB) {
        window.open(REDIRECT_URL, "_blank", "noopener,noreferrer");
      } else {
        window.location.replace(REDIRECT_URL);
      }
    }, 120);
  };

  const validatePage = (page) => {
    const newErrors = {};
    if (page === 1) {
      if (!formData.age) newErrors.age = "Age is required.";
      if (!formData.gender) newErrors.gender = "Gender is required.";
      if (formData.gender === "Other" && !formData.genderOther)
        newErrors.genderOther = "Please specify your gender.";
      if (!formData.race) newErrors.race = "Race is required.";
      if (formData.race === "Other" && !formData.raceOther)
        newErrors.raceOther = "Please specify your race.";
      if (!formData.education) newErrors.education = "Education level is required.";
      if (formData.education === "Other" && !formData.educationOther)
        newErrors.educationOther = "Please specify your education level.";
      if (!formData.country) newErrors.country = "Country/Region is required.";
      if (!formData.politicalOrientation)
        newErrors.politicalOrientation = "Political orientation is required.";
    } else if (page === 2) {
      // NOW: Universities & Hiring page
      if (!formData.familiarityUniversities?.length)
        newErrors.familiarityUniversities = "Select at least one option.";
      if (!formData.employmentStatus)
        newErrors.employmentStatus = "Employment status is required.";
      if (formData.employmentStatus === "Other" && !formData.employmentStatusOther)
        newErrors.employmentStatusOther = "Please specify your employment status.";
      if (!formData.hiringInvolved)
        newErrors.hiringInvolved = "Please answer this question.";
      if (formData.hiringInvolved === "Yes" && !formData.hiringExperienceYears)
        newErrors.hiringExperienceYears = "Please select your approximate years of experience.";
    } else if (page === 3) {
      // NOW: Game Experience page
      if (!formData.experience) newErrors.experience = "This question is required.";
      if (formData.experience === "Yes" && !formData.experienceDetails)
        newErrors.experienceDetails = "Please provide details about your experience.";
      if (!formData.motivation) newErrors.motivation = "Motivation is required.";
      if (!formData.engagement)
        newErrors.engagement = "Please rate the engagement of the game.";
      if (!formData.clearity)
        newErrors.clearity = "Please rate the clarity of the instructions.";
      if (!formData.interferenceOption)
        newErrors.interferenceOption = "Please indicate if you encountered any technical issues.";
      if (formData.interferenceOption === "Yes" && !formData.interference)
        newErrors.interference = "Please describe the technical issues or distractions.";
      if (!formData.keyFactors)
        newErrors.keyFactors = "Please describe the key factors that influenced your decisions.";
      if (!formData.strategy)
        newErrors.strategy = "Please explain your decision-making strategy.";
      if (!formData.followUp)
        newErrors.followUp = "Please indicate if you would like to participate in follow-up studies.";
    }
    return newErrors;
  };

  return (
    <div className="py-8 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Page 1 */}
      {currentPage === 1 && (
        <form onSubmit={(e) => e.preventDefault()}>
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4 mt-6">
            Exit Survey (Page 1 of 4)
          </h3>
          <p className="text-gray-700 text-sm mb-6">
            <strong>If you didn’t play or join the game</strong>, please just <strong>complete the exit survey</strong>. Thanks!
          </p>

          <div className="space-y-6">
            {/* Age */}
            <div>
              <label htmlFor="age" className="block text-sm font-medium text-gray-700 mb-1">
                Age
              </label>
              <input
                id="age"
                name="age"
                type="number"
                autoComplete="off"
                min="1"
                max="100"
                className="appearance-none block w-24 px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-empirica-500 focus:border-empirica-500 sm:text-sm"
                value={formData.age || ""}
                onChange={(e) => {
                  const value = parseInt(e.target.value, 10);
                  if (value >= 1 && value <= 100) {
                    handleInputChange("age", value);
                  } else if (e.target.value === "") {
                    handleInputChange("age", "");
                  }
                }}
              />
              {errors.age && (
                <p className="text-red-500 text-sm mt-1">{errors.age}</p>
              )}
            </div>

            {/* Gender */}
            <div>
              <label className={labelClassName}>Gender</label>
              <div className="grid gap-2">
                {["Female", "Male", "Prefer not to say", "Other"].map((option) => (
                  <Radio
                    key={option}
                    selected={formData.gender}
                    name="gender"
                    value={option}
                    label={option}
                    onChange={(e) => handleInputChange("gender", e.target.value)}
                  />
                ))}
                {formData.gender === "Other" && (
                  <input
                    type="text"
                    className={inputClassName}
                    placeholder="Please specify"
                    value={formData.genderOther}
                    onChange={(e) => handleInputChange("genderOther", e.target.value)}
                  />
                )}
                {errors.gender && <div className="text-red-500 text-sm">{errors.gender}</div>}
                {errors.genderOther && <div className="text-red-500 text-sm">{errors.genderOther}</div>}
              </div>
            </div>

            {/* Race */}
            <div>
              <label className={labelClassName}>Race</label>
              <div className="grid gap-2">
                {["African", "Asian", "Caucasian", "Latin/x", "Native American", "Mixed-Race", "Prefer not to say", "Other"].map((option) => (
                  <Radio
                    key={option}
                    selected={formData.race}
                    name="race"
                    value={option}
                    label={option}
                    onChange={(e) => handleInputChange("race", e.target.value)}
                  />
                ))}
                {formData.race === "Other" && (
                  <input
                    type="text"
                    className={inputClassName}
                    placeholder="Please specify"
                    value={formData.raceOther}
                    onChange={(e) => handleInputChange("raceOther", e.target.value)}
                  />
                )}
                {errors.race && <div className="text-red-500 text-sm">{errors.race}</div>}
                {errors.raceOther && <div className="text-red-500 text-sm">{errors.raceOther}</div>}
              </div>
            </div>

            {/* Education */}
            <div>
              <label className={labelClassName}>Highest Education Qualification</label>
              <div className="grid gap-2">
                {[
                  "Did not graduate from high school",
                  "High School",
                  "Some College",
                  "College",
                  "Graduate Professional School",
                  "Prefer not to say",
                  "Other",
                ].map((option) => (
                  <Radio
                    key={option}
                    selected={formData.education}
                    name="education"
                    value={option}
                    label={option}
                    onChange={(e) => handleInputChange("education", e.target.value)}
                  />
                ))}
                {formData.education === "Other" && (
                  <input
                    type="text"
                    className={inputClassName}
                    placeholder="Please specify"
                    value={formData.educationOther}
                    onChange={(e) => handleInputChange("educationOther", e.target.value)}
                  />
                )}
                {errors.education && <div className="text-red-500 text-sm">{errors.education}</div>}
                {errors.educationOther && <div className="text-red-500 text-sm">{errors.educationOther}</div>}
              </div>
            </div>

            {/* Country */}
            <div>
              <label className={labelClassName}>Primary Country/Region of Residence</label>
              <input
                id="country"
                name="country"
                autoComplete="off"
                className={inputClassName}
                value={formData.country}
                onChange={(e) => handleInputChange("country", e.target.value)}
              />
              {errors.country && <div className="text-red-500 text-sm">{errors.country}</div>}
            </div>

            {/* Political Orientation */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Political Orientation
              </label>
              <div className="grid gap-2">
                {[
                  "Extremely Conservative",
                  "Moderately Conservative",
                  "Slightly Conservative",
                  "Slightly Liberal",
                  "Moderately Liberal",
                  "Extremely Liberal",
                  "Prefer not to say",
                ].map((option) => (
                  <Radio
                    key={option}
                    selected={formData.politicalOrientation}
                    name="politicalOrientation"
                    value={option}
                    label={option}
                    onChange={(e) => handleInputChange("politicalOrientation", e.target.value)}
                  />
                ))}
              </div>
              {errors.politicalOrientation && <div className="text-red-500 text-sm">{errors.politicalOrientation}</div>}
            </div>
          </div>

          <div className="mt-6">
            <Button type="button" className="bg-blue-500 text-white" handleClick={handleNextPage}>
              Next Page
            </Button>
          </div>
        </form>
      )}

      {/* Page 2 (NOW: Universities & Hiring) */}
      {currentPage === 2 && (
        <form onSubmit={(e) => e.preventDefault()}>
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-2">
            Exit Survey (Page 2 of 4)
          </h3>

          <div className="space-y-8">
            {/* Familiarity with Universities */}
            <div>
              <p className="text-gray-700 text-lg font-bold mb-2">
                We would like to ask your prior familiarity with universities:
              </p>
              <label className={labelClassName}>
                Before this study, which of the following universities had you heard of?{" "}
                <span className="font-normal">(Select all that apply)</span>
              </label>
              <div className="grid gap-2">
                {UNIVERSITY_OPTIONS.map((u) => (
                  <Checkbox
                    key={u}
                    name="familiarityUniversities"
                    value={u}
                    label={u}
                    checked={formData.familiarityUniversities.includes(u)}
                    onChange={() => toggleUniversity(u)}
                  />
                ))}
              </div>
              {errors.familiarityUniversities && (
                <div className="text-red-500 text-sm mt-1">{errors.familiarityUniversities}</div>
              )}
            </div>

            {/* Work and Hiring Experience */}
            <div>
              <p className="text-gray-700 text-lg font-bold mb-2">
                We would like to ask your work and hiring experience
              </p>

              {/* Employment status */}
              <div className="mb-4">
                <label className={labelClassName}>
                  What best describes your employment status over the last three months?
                </label>
                <div className="grid gap-2">
                  {EMPLOYMENT_OPTIONS.map((opt) => (
                    <Radio
                      key={opt}
                      selected={formData.employmentStatus}
                      name="employmentStatus"
                      value={opt}
                      label={opt}
                      onChange={(e) => handleInputChange("employmentStatus", e.target.value)}
                    />
                  ))}
                </div>
                {formData.employmentStatus === "Other" && (
                  <input
                    type="text"
                    className={`${inputClassName} mt-2`}
                    placeholder="Please specify"
                    value={formData.employmentStatusOther}
                    onChange={(e) => handleInputChange("employmentStatusOther", e.target.value)}
                  />
                )}
                {errors.employmentStatus && (
                  <div className="text-red-500 text-sm">{errors.employmentStatus}</div>
                )}
                {errors.employmentStatusOther && (
                  <div className="text-red-500 text-sm">{errors.employmentStatusOther}</div>
                )}
              </div>

              {/* Hiring involvement */}
              <div className="mb-4">
                <label className={labelClassName}>
                  Does your current or past job involve evaluating, selecting, or hiring people?
                </label>
                <div className="grid gap-2">
                  <Radio
                    selected={formData.hiringInvolved}
                    name="hiringInvolved"
                    value="Yes"
                    label="Yes"
                    onChange={(e) => handleInputChange("hiringInvolved", e.target.value)}
                  />
                  <Radio
                    selected={formData.hiringInvolved}
                    name="hiringInvolved"
                    value="No"
                    label="No"
                    onChange={(e) => handleInputChange("hiringInvolved", e.target.value)}
                  />
                </div>
                {errors.hiringInvolved && (
                  <div className="text-red-500 text-sm">{errors.hiringInvolved}</div>
                )}
              </div>

              {/* Years of experience (conditional) */}
              {formData.hiringInvolved === "Yes" && (
                <div className="mb-4">
                  <label className={labelClassName}>
                    Approximately how many years of experience do you have in roles involving hiring or evaluation decisions?
                  </label>
                  <div className="grid gap-2">
                    {HIRING_YEARS.map((opt) => (
                      <Radio
                        key={opt}
                        selected={formData.hiringExperienceYears}
                        name="hiringExperienceYears"
                        value={opt}
                        label={opt}
                        onChange={(e) =>
                          handleInputChange("hiringExperienceYears", e.target.value)
                        }
                      />
                    ))}
                  </div>
                  {errors.hiringExperienceYears && (
                    <div className="text-red-500 text-sm">{errors.hiringExperienceYears}</div>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 flex justify-between">
            <Button type="button" className="bg-gray-500 text-white" handleClick={handlePreviousPage}>
              Previous Page
            </Button>
            <Button type="button" className="bg-blue-500 text-white" handleClick={handleNextPage}>
              Next Page
            </Button>
          </div>
        </form>
      )}

      {/* Page 3 (NOW: Game Experience) */}
      {currentPage === 3 && (
        <form onSubmit={(e) => e.preventDefault()}>
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
            Exit Survey (Page 3 of 4)
          </h3>
          <p className="text-gray-700 text-lg font-bold mb-6">
            We would like to ask your experiences with this game:
          </p>
          <div className="space-y-6">
            {/* Experience */}
            <div>
              <label className={labelClassName}>
                Have you participated in similar experiments before?
              </label>
              <div className="grid gap-2">
                <Radio
                  selected={formData.experience}
                  name="experience"
                  value="Yes"
                  label="Yes"
                  onChange={(e) => handleInputChange("experience", e.target.value)}
                />
                <Radio
                  selected={formData.experience}
                  name="experience"
                  value="No"
                  label="No"
                  onChange={(e) => handleInputChange("experience", e.target.value)}
                />
              </div>
              {formData.experience === "Yes" && (
                <textarea
                  id="experienceDetails"
                  name="experienceDetails"
                  rows="4"
                  className={`${inputClassName} mt-4`}
                  placeholder="Please share details such as when, where, who organized it (e.g., university, company, etc.), and a brief description of the experiment, if possible."
                  value={formData.experienceDetails || ""}
                  onChange={(e) => handleInputChange("experienceDetails", e.target.value)}
                />
              )}
              {errors.experience && (
                <div className="text-red-500 text-sm">{errors.experience}</div>
              )}
              {formData.experience === "Yes" && errors.experienceDetails && (
                <div className="text-red-500 text-sm">{errors.experienceDetails}</div>
              )}
            </div>

            {/* Follow-Up Studies */}
            <div>
              <label className={labelClassName}>
                Would you like to participate in follow-up studies?
              </label>
              <div className="grid gap-2">
                <Radio
                  selected={formData.followUp}
                  name="followUp"
                  value="Yes"
                  label="Yes"
                  onChange={(e) => handleInputChange("followUp", e.target.value)}
                />
                <Radio
                  selected={formData.followUp}
                  name="followUp"
                  value="No"
                  label="No"
                  onChange={(e) => handleInputChange("followUp", e.target.value)}
                />
              </div>
              {errors.followUp && <div className="text-red-500 text-sm">{errors.followUp}</div>}
            </div>

            {/* Engagement */}
            <div className="mb-4">
              <label className={labelClassName}>How engaging was the game?</label>
              <div className="grid gap-2">
                {[
                  "Not Engaging",
                  "Slightly Engaging",
                  "Moderately Engaging",
                  "Engaging",
                  "Very Engaging",
                ].map((option) => (
                  <Radio
                    key={option}
                    selected={formData.engagement}
                    name="engagement"
                    value={option}
                    label={option}
                    onChange={(e) => handleInputChange("engagement", e.target.value)}
                  />
                ))}
              </div>
              {errors.engagement && (
                <p className="text-red-500 text-sm mt-1">{errors.engagement}</p>
              )}
            </div>

            {/* Task Clarity */}
            <div className="mb-4">
              <label className={labelClassName}>How clear were the instructions?</label>
              <div className="grid gap-2">
                {[
                  "Very Unclear",
                  "Slightly Clear",
                  "Moderately Clear",
                  "Clear",
                  "Very Clear",
                ].map((option) => (
                  <Radio
                    key={option}
                    selected={formData.clearity}
                    name="clearity"
                    value={option}
                    label={option}
                    onChange={(e) => handleInputChange("clearity", e.target.value)}
                  />
                ))}
              </div>
              {errors.clearity && (
                <p className="text-red-500 text-sm mt-1">{errors.clearity}</p>
              )}
            </div>

            {/* Decision-Making */}
            <div>
              <label className={labelClassName}>
                How did you approach decision-making in this experiment?
              </label>
              <textarea
                id="strategy"
                name="strategy"
                rows="4"
                className={inputClassName}
                placeholder="Describe your approach to making decisions during the experiment..."
                value={formData.strategy}
                onChange={(e) => handleInputChange("strategy", e.target.value)}
              />
              {errors.strategy && (
                <div className="text-red-500 text-sm">{errors.strategy}</div>
              )}
            </div>

            {/* Key Factors */}
            <div>
              <label className={labelClassName}>
                What key factors influenced your decisions?
              </label>
              <textarea
                id="keyFactors"
                name="keyFactors"
                rows="4"
                className={inputClassName}
                placeholder="Please highlight the main factors that guided your choices..."
                value={formData.keyFactors || ""}
                onChange={(e) => handleInputChange("keyFactors", e.target.value)}
              />
              {errors.keyFactors && (
                <div className="text-red-500 text-sm">{errors.keyFactors}</div>
              )}
            </div>

            {/* Motivation */}
            <div>
              <label className={labelClassName}>What motivated you to participate in this experiment?</label>
              <textarea
                id="motivation"
                name="motivation"
                rows="4"
                className={inputClassName}
                placeholder="Please share your motivation for participating..."
                value={formData.motivation}
                onChange={(e) => handleInputChange("motivation", e.target.value)}
              />
              {errors.motivation && <div className="text-red-500 text-sm">{errors.motivation}</div>}
            </div>

            {/* Technical Issues */}
            <div>
              <label className={labelClassName}>
                Did you encounter any technical issues or distractions?
              </label>
              <div className="grid gap-2">
                <Radio
                  selected={formData.interferenceOption}
                  name="interferenceOption"
                  value="Yes"
                  label="Yes"
                  onChange={(e) =>
                    handleInputChange("interferenceOption", e.target.value)
                  }
                />
                <Radio
                  selected={formData.interferenceOption}
                  name="interferenceOption"
                  value="No"
                  label="No"
                  onChange={(e) =>
                    handleInputChange("interferenceOption", e.target.value)
                  }
                />
              </div>
              {formData.interferenceOption === "Yes" && (
                <textarea
                  id="interference"
                  name="interference"
                  rows="4"
                  className={`${inputClassName} mt-4`}
                  placeholder="Please describe the issue..."
                  value={formData.interference || ""}
                  onChange={(e) =>
                    handleInputChange("interference", e.target.value)
                  }
                />
              )}
              {errors.interferenceOption && (
                <div className="text-red-500 text-sm">{errors.interferenceOption}</div>
              )}
              {formData.interferenceOption === "Yes" &&
                errors.interference && (
                  <div className="text-red-500 text-sm">{errors.interference}</div>
                )}
            </div>

            {/* Additional Feedback */}
            <div>
              <label className={labelClassName}>
                Any additional thoughts or feedback you'd like to share? (Optional)
              </label>
              <textarea
                id="additionalFeedback"
                name="additionalFeedback"
                rows="4"
                className={inputClassName}
                placeholder="Feel free to share any other comments..."
                value={formData.additionalFeedback || ""}
                onChange={(e) => handleInputChange("additionalFeedback", e.target.value)}
              />
            </div>
          </div>

          <div className="mt-6 flex justify-between">
            <Button type="button" className="bg-gray-500 text-white" handleClick={handlePreviousPage}>
              Previous Page
            </Button>
            <Button type="button" className="bg-blue-500 text-white" handleClick={handleNextPage}>
              Next Page
            </Button>
          </div>
        </form>
      )}

      {/* Page 4 */}
      {currentPage === 4 && (
        <form onSubmit={handleSubmit}>
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
            Exit Survey (Page 4 of 4)
          </h3>
          <Alert title="Bonus">
            {/* <p>
              Please click <strong>Submit</strong> to finish the survey and submit the following code to receive your bonus:{" "}
              <strong>{player.id}</strong>.
            </p> */}
            <p>
              Please click <strong>Submit</strong> to finish the survey and redriect to get your rewards.
            </p>
            {/* <p className="pt-1">
              Reminder: Save it securely to claim your reward.
            </p> */}
            <p className="pt-1">
              If you enjoyed the game, we’d really appreciate a <strong>5-star rating</strong>! ⭐⭐⭐⭐⭐
            </p>
            <p className="pt-1">
              Thank you so much for participating!
            </p>
          </Alert>
          <div className="mt-6 flex justify-between">
            <Button type="button" className="bg-gray-500 text-white" handleClick={handlePreviousPage}>
              Previous Page
            </Button>
            <Button type="submit" className="bg-blue-500 text-white">
              Submit
            </Button>
          </div>
        </form>
      )}
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

export function Checkbox({ name, value, label, checked, onChange }) {
  return (
    <label className="text-sm font-medium text-gray-700">
      <input
        className="mr-2 shadow-sm sm:text-sm"
        type="checkbox"
        name={name}
        value={value}
        checked={checked}
        onChange={onChange}
      />
      {label}
    </label>
  );
}
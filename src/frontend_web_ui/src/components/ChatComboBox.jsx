import { Combobox, ComboboxLabel, ComboboxOption } from '@/components/catalyst/combobox'
import { Field, Label } from '@/components/catalyst/fieldset'

export default function({ disabled = false, htmlName, htmlLabel, selectedOption, opts, setState }) {
  return (
    <Field>
      <Label>{htmlLabel}</Label>
      <Combobox
        name={htmlName}
        options={opts}
        displayValue={(option) => option}
        defaultValue={selectedOption}
        onChange={setState}
        disabled={disabled}
      >
        {(option) => (
          <ComboboxOption value={option}>
            <ComboboxLabel>{option}</ComboboxLabel>
          </ComboboxOption>
        )}
      </Combobox>
    </Field>
  )
}
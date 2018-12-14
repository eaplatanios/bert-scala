/* Copyright 2018, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.bert.data.tokenization

import java.text.Normalizer

import scala.util.matching.Regex

/** Runs basic tokenization (punctuation splitting, lower casing, etc.).
  *
  * @param  vocabulary    Vocabulary to use.
  * @param  caseSensitive Specifies whether or not to ignore case.
  *
  * @author Emmanouil Antonios Platanios
  */
class BasicTokenizer(
    override val vocabulary: Vocabulary,
    val caseSensitive: Boolean = false
) extends Tokenizer {
  /** Tokenizes a piece of text. */
  override def tokenize(text: String): Seq[String] = {
    val cleaned = clean(text)
    val tokenized = Tokenizer.whitespaceRegex.split(cleaned)
    tokenized.flatMap(token => {
      var processed = token
      if (!caseSensitive) {
        processed = processed.toLowerCase()

        // Normalize unicode characters.
        processed = Normalizer.normalize(processed, Normalizer.Form.NFD)

        // Strip accents.
        processed = BasicTokenizer.nonSpacingMarkRegex.replaceAllIn(processed, "")
      }

      // Split punctuation.
      processed = BasicTokenizer.punctuationRegex.replaceAllIn(processed, " $1 ")

      Tokenizer.splitOnWhitespace(processed)
    })
  }

  /** Performs invalid character removal and whitespace cleanup on the provided text. */
  def clean(text: String): String = {
    // Normalize whitespaces.
    val afterWhitespace = Tokenizer.whitespaceRegex.replaceAllIn(text, " ")

    // Remove control characters.
    val afterControl = BasicTokenizer.controlRegex.replaceAllIn(afterWhitespace, "")

    // Add whitespace around CJK characters.
    val afterCJP = BasicTokenizer.cjpRegex.replaceAllIn(afterControl, " $1 ")
    afterCJP
  }
}

object BasicTokenizer {
  private val controlRegex       : Regex = "[\\x{0000}\\x{fffd}\\p{C}]".r
  private val nonSpacingMarkRegex: Regex = "\\p{Mn}".r

  /** Regular expression matching punctuation characters.
    *
    * We treat all non-letter/number ASCII as punctuation. Characters such as "$" are not in the Unicode
    * Punctuation class but we treat them as punctuation anyways, for consistency.
    */
  private val punctuationRegex: Regex = {
    "([\\p{P}!-/:-@\\[-`{-~])".r
  }

  /** Regular expression matching CJK characters.
    *
    * This regular expression defines a "chinese character" as anything in the
    * [CJK Unicode block](https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)).
    *
    * Note that the CJK Unicode block is not all Japanese and Korean characters, despite its name.
    * The modern Korean Hangul alphabet is a different block, as is Japanese Hiragana and Katakana.
    * Those alphabets are used to write space-separated words, and so they are not treated specially
    * and are instead handled like all of the other languages.
    */
  private val cjpRegex: Regex = {
    ("([\\p{InCJK_Unified_Ideographs}" +
        "\\p{InCJK_Unified_Ideographs_Extension_A}" +
        "\\p{InCJK_Compatibility_Ideographs}" +
        "\\x{20000}-\\x{2a6df}" +
        "\\x{2a700}-\\x{2b73f}" +
        "\\x{2b740}-\\x{2b81f}" +
        "\\x{2b820}-\\x{2ceaf}" +
        "\\x{2f800}-\\x{2fa1f}])").r
  }
}

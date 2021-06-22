# This Python file uses the following encoding: utf-8
## From mck packages
## -- developed by Miroslav Fil

import csv
import re
import string
import numpy as np
import pandas as pd
import datetime
import unidecode
import nltk
import os
import itertools
import math
from multiprocess import Pool, cpu_count
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from urllib.parse import urlparse


class EmailCleaning:
    @staticmethod
    def full_clean(email):
        """Used in EIG. For other use-cases, it might be better to compose individual parts. This is end-to-end cleaning pipeline

        Args:
            email (Str): Text to be cleaned. It is expected to be full of escapes such as \n, \t,..

        Returns:
            email (Str): Cleaned up email
        """
        email = EmailCleaning.remove_names(email, methods=['regex'])
        email = EmailCleaning.remove_repeated_replies(email)
        email = EmailCleaning.remove_greetings(email)
        email = EmailCleaning.fix_whitespace_formatting(email)
        email = EmailCleaning.remove_email_metadata(email)
        email = EmailCleaning.remove_signatures(email)
        email = EmailCleaning.clean_fixed_terms(email)
        email = EmailCleaning.anonymize_urls(email)
        email = EmailCleaning.anonymize_files(email)
        email = EmailCleaning.anonymize_email_adresses(email)
        email = EmailCleaning.clean_redundant_new_lines(email)
        email = EmailCleaning.clean_single_leading_newline(email)
        email = EmailCleaning.collapse_multiple_spaces(email)
        # this one depends on previous cleaning of paragraphs (/n/n), might delete more than you would like
        email = EmailCleaning.remove_repeating_parags(email)
        return email

    @staticmethod
    def parallelize_cleaning(ordered_iterable, cleaning_fun=full_clean,
                             num_of_processes=math.ceil((cpu_count() / 2) - 1), wrapper='pd.apply'):
        """A wrapper to parallelize pandas.apply. Meant for data parallelism

        Args:
            ordered_iterable (Ordered iterable): Typically a DataFrame. Should support splitting into n parts
            cleaning_fun ([type], optional): [description]. Defaults to full_clean. Needs to be embarassingly parallel
            num_of_processes ([type], optional): [description]. Defaults to math.ceil((cpu_count()/2)-1). It should match number of physical cores
            wrapper (str, optional): [description]. Defaults to 'pd.apply'. Whether to use it in pd.apply or as standalone (for example, when not operating on DataFrames)

        Returns:
            transformed_iterable: The cleaned up ordered_iterable returned in the same shape
        """

        def pandas_wrapper(fun):
            return lambda x: x.apply(fun)

        ordered_iterable_split = np.array_split(ordered_iterable, num_of_processes)
        pool = Pool(num_of_processes)
        if wrapper == 'pd.apply':
            cleaning_fun = pandas_wrapper(cleaning_fun)
            transformed_data = pd.concat(pool.map(cleaning_fun, ordered_iterable_split))
        if wrapper == 'none':
            transformed_data = itertools.chain(pool.map(cleaning_fun, ordered_iterable_split))
        pool.close()
        pool.join()
        return transformed_data

    @staticmethod
    def fix_whitespace_formatting(email):
        regex = []
        # Merges multiple normal spaces into one, "fox  is dog" -> "fox is dog"
        regex.append([' {2,}', ' '])
        # remove line dividers including spaces
        regex.append([r'^[^\nA-Za-z0-9]*[^\nA-Za-z0-9]$', ''])
        # remove tab and '>'
        regex.append([r'^(\t|\>?)+', ''])
        # remove leading spaces
        regex.append([r'^[^\n]\s+[^\S]', '\n'])
        # clear extra lines, \n\n\n\t\t\n\n\n\n -> \n\n
        regex.append([r'(\s*[\n]\s*){3,30}', '\n\n'])
        regex.append([r'\n{1}>[>\s]+', '\n'])
        for rg in regex:
            email = re.sub(rg[0], rg[1], email, flags=re.IGNORECASE | re.MULTILINE)
        return email

    @staticmethod
    def remove_email_metadata(email):
        """Removes email metadata such as From:, Re:,.. It is quite similar to remove_repeated_replies as is. TODO differentiate them
        DANGEROUS: Most regexes match from first such occurence until the end of the string. This might delete a lot of your stuff if its not your intention

        Args:
            email (Str): Email to be cleaned up

        Returns:
            email (Str): Cleaned up email
        """
        # recipients
        regex = []
        regex_cased = []
        regex.append([r'To: [0-9a-zA-Z-_:@\.;<>\s\w]+Subject:\s*', ' '])
        # remove email and other metadata
        # this also deletes stuff like "on both"
        regex_cased.append([r'On [0-9a-zA-Z.\s]+ at [0-9]+\:[0-9]{2} [a-zA-Z0-9@_\.\s<]+>\s+wrote:', ' '])
        regex_cased.append([r'Am [0-9a-zA-Z.-/\s]+ um [0-9]+\:[0-9]{2} schrieb\s+[a-zA-Z0-9@_\.\s<]+>:', ' '])
        regex_cased.append([
                               r'(\s|^|\|)(Received from|TO|From|Date|Sent by|sent|CC|BCC|Cc|Bcc|cc|bcc|Copy To|Sender|Deliver to|Recipient Name|Period|sent email|Forwarded At)[\s\S]*?:.*',
                               ' '])
        # remove Forwarded by
        regex.append([r'----- Forwarded (by)?.+(\n.*)?\s(-----|\d\d\:\d\d\s((P|A)M)?)', ' '])
        # #various useless metadata
        regex.append([
                         r'^(user|owner|Timezone|FMNO|name|ip|date|time|path|agent|Code|Domain|address|room|Location|Team|Responder|Contact|Office|Telefono|Tel|fax|Mobile|Call Number|cell|callback|direct|voip|Attn|Colleague|phone|Department|Telephone|RSA|Organization|Country|Group|Email|Assistant|EA|Region|Database|view|Tracking|Server|path|Hours|Reference\s?\#|Created|Tracking|t|e|m|f)\s*?:.*?$',
                         ' '])
        # various useless metadata - partial text
        regex.append([r'^[a-z0-9]+\s(name|ip|date|time|path|agent|Code|Domain)[a-z0-9]*\s*?:.*?$', ' '])
        # out of the office
        regex.append([r'AUTO:.+out of(\sthe)? office.*(\n.+)*', ''])
        # tables |xxxxx|
        regex.append([r'^.*?\|.*?$', ' '])
        # telephone numbers, ..
        regex.append([r'[-_+;:.,\s]+[0-9]+[0-9-_+;:.,\s]*(?:$|\n)', '\n'])
        for rg in regex:
            email = re.sub(rg[0], rg[1], email, flags=re.IGNORECASE | re.MULTILINE)
        for rg in regex_cased:
            email = re.sub(rg[0], rg[1], email, flags=re.MULTILINE)
        return email

    @staticmethod
    def remove_signatures(email):
        """Removes signatures. Strict

        Args:
            email (Str): Email to be cleaned up

        Returns:
            email (Str): Cleaned up email
        """
        regex = []
        regex.append([
                         r'^(McKinsey & Company|McKinsey & Co.|Cheers|Sincerely|Best (regards)?|Many thanks|kind regards|Regards|Warm regards|tnx|thx|Thank you|Thanks (?!to)|Assistant|Assistant:|Executive Assistant:).*?(\n){1,3}\s?(([A-Z][a-z]{3,20}\s?){1,3})',
                         ' '])
        regex.append([
                         r'^(McKinsey & Company|Cheers|Sincerely|Best (regards)?|Many thanks|kind regards|Regards|Warm regards|thanks|tnx|thx|Thank you|Thanks (?!to)Assistant|Assistant:|Executive Assistant:).*?[!\w\s$]*',
                         ' '])
        regex.append([r'^\s*(Re:|Fw:|Subject:).*?(\n){1,3}', '\n'])
        # remove signatures by phone number and '|'
        regex.append([r'(\n.+){0,5}\+\0{0,2}\s?\d{1,3}\s?\-?\(?\d{1,3}\)?(\s?\-?\d{1,4}){1,5}\s?\|(.+\n){0,5}', '\n'])
        # remove autogenerated email
        regex.append([r'Voice message from.*?\d\d\:\d\d\s((P|A)M)?\n(.*\n)+?.+contact the Global.*$', ' '])
        # remove ghd and CC signatures
        regex.append([r'Global Helpdesk\n.+313(\n.+){2,3}', ''])
        regex.append([r'^(Customer Care|IT Customer Experience)\n(.[^\n]+\n){1,4}', ' '])
        for rg in regex:
            email = re.sub(rg[0], rg[1], email, flags=re.IGNORECASE | re.MULTILINE)
        return email

    @staticmethod
    def remove_greetings(email):
        """Heuristically matches short greetings (eg. it tries to work with word count, potential newlines,..)

        Args:
            email (Str): Email to be cleaned up

        Returns:
            email (Str): Cleaned up email
        """
        # deletes short greetings that are <20 chars
        regex = []
        # remove greetings
        regex.append([
                         r'^(Folks|All|Hej|hey|hi|hello|good|dear|friends|team|Good day|Greetings)(\s|,|\.|\n|\!)(([a-z]{3,20}\s?\/?){0,3})(\.|-|,|;|\s)?(\\n)?\n\n',
                         ''])
        for rg in regex:
            email = re.sub(rg[0], rg[1], email, flags=re.IGNORECASE | re.MULTILINE)
        return email

    @staticmethod
    def clean_fixed_terms(email):
        """Deletes fixed trash terms by direct match

        Args:
            email (Str): Email to be cleaned up

        Returns:
            email (Str): Cleaned up email
        """
        useless_ngrams = ["mckinsey\s+company", "best\s+regards", "kind\s+regards", "thank\s+you",
                          "sent from my iphone", "Von meinem iPhone gesendet", "pacific\s+time",
                          'This email is confidential and may be privileged(.+\n)+.+\suse it for any purpose.',
                          'Removed.*?can be found in Emails', '[a-z]+\scall was linked to the incident',
                          'Knowledge article KO.*\:\n.*$']
        # months do not include may because it matches with the verb
        useless_words = ["folks", "hi", "hello", "dear", "sincerely",
                         "friends", "please", "tnx", "thanks", "fw", "re", "fwd", "january", "jan", "february", "feb",
                         "march", "mar", "april",
                         "apr", "june", "jun", "july", "jul", "august", "aug", "september", "sep", "october", "oct",
                         "november", "nov", "december", "dec",
                         "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "pm", "am",
                         "est", "today", "tomorrow", "yesterday"]
        for useless_ngram in useless_ngrams:
            email = re.sub(useless_ngram, ' ', email, flags=re.IGNORECASE | re.MULTILINE)
        # could also use regex to remove direct words but would be harder for no benefit
        # email.split without any argument would also strip newlines in \ncompendium
        admissible_words = [w for w in email.split(" ") if not w in useless_words]
        email = ' '.join(admissible_words)
        return email

    @staticmethod
    def remove_short_lines(email, threshold=35, threshold_with_punctuation=20):
        """Heuristically removes short lines (delimited by \n) because they are likely to be greetings, farewells, ..

        Args:
            email (Str): Email to be cleaned up
            threshold (int, optional): Maximum char length for deletion if the line does not contain punctuation. Defaults to 35.
            threshold_with_punctuation (int, optional): Maximum char length for deletion if the line containts punctuation (=assumed to be more likely to have good content). Defaults to 20.

        Returns:
            new_email(str): Cleaned up email
        """

        # need to be careful about applying this because emails have line delimiters
        # by nature (be it as normal text or from lists of items..) and even single words in sentences might then get deleted..
        def longer_than(line, lnt):
            value = len(
                ''.join([i for i in line if not i.isdigit() and i != ' ']).replace('ANONYMIZED_NAME', '').replace(
                    'ANONYMIZED_LINK', '').replace('ANONYMIZED_FILE', '')) > lnt
            return value

        def ends_with_punctuation(line):
            value = ''.join([i for i in line if not i.isdigit() and i != ' ']).replace('ANONYMIZED_NAME', '').replace(
                'ANONYMIZED_LINK', '').replace('d', '')[-1] in ".,;:!?-"
            return value

        new_email = ""
        paragraphs = re.split(r'\n{2,}', email)
        for p in paragraphs:
            lines = p.split('\n')
            new_paragraph = '\n'.join([line for line in lines if longer_than(line, threshold) or (
                        longer_than(line, threshold_with_punctuation) and ends_with_punctuation(line))])
            new_email = new_email + "\n\n" + new_paragraph
        return new_email

    @staticmethod
    def clean_redundant_new_lines(email):
        """Collapses high number of \n to just \n\n within a line and deletes all at beginning\start of line

        Args:
            email (Str): Email to be cleaned up

        Returns:
            email (Str): Cleaned up mail
        """
        # removes >2 newlines at start of line
        email = re.sub(r'^\n{2,}', r'', email)
        # reduces >2 newlines to just two
        email = re.sub(r'\n{2,}', r'\n\n', email)
        # removes >2 newlines at end of line
        email = re.sub(r'\n{2,}$', r'', email)
        return email

    @staticmethod
    def clean_multiple_leading_whitespaces(email):
        """Cleans up whitespace trash in front of words aggresively.

        #\ncompendium ->' compendium', but \t\t \n compendium -> ' compendium' also

        Args:
            email (Str): Email to be cleaned up

        Returns:
            email (Str): Cleaned up email
        """
        email = [w for w in email.split()]
        email = ' '.join(email)
        return email

    @staticmethod
    def clean_single_leading_newline(email, strict=True):
        """Cleans up newlines like \nbananas -> " bananas". This might create doubled whitespaces (but it might also not, if the \n are making up a list of items)
        DANGEROUS: Might interact badly with list of items separated by \n only
        CAREFUL! This regex wont work in online regex editors (there it would be (?<!\)\\n(?=\w)) but it will work when you run and vice versa

        Args:
            email (Str): Email to be cleaned up
            strict (bool, optional): If True, will not delete \n followed by capital character. It's more likely to not wrongy delete lists of items. Defaults to True.

        Returns:
            email (Str): Cleaned up email
        """
        # Think about whether you even want this, this is probably not good if in your text, people use \n to format the text, especially in lists of items.
        regex = []
        if strict == True:
            # Won't match capital chars (\nJames) - useful to deal with lists of items, but fails on names..
            regex.append([r'(?<!\n)\n(?=[a-z0-9])', ' '])
        else:
            regex.append([r'(?<!\n)\n(?=\w)', ' '])
        for rg in regex:
            email = re.sub(rg[0], rg[1], email, flags=re.MULTILINE)
        return email

    @staticmethod
    def collapse_multiple_spaces(email):
        """what   is this   shit" -> "what is this shit", wont work on whitespaces like /t etc.
        If desired, you need to replace " " in the regex with "\s"

        Args:
            email (Str): Email to be cleaned up

        Returns:
            email (Str): Cleaned up email
        """
        regex = []
        # "what   is this   shit" -> "what is this shit", wont work on whitespaces like /t etc.
        regex.append([' {2,}', ' '])
        for rg in regex:
            email = re.sub(rg[0], rg[1], email, flags=re.IGNORECASE | re.MULTILINE)
        return email

    @staticmethod
    def remove_names(email, methods, replacement='ANONYMIZED_NAME'):
        """Anonymize names (of both companies and people), replacing them with placeholder.
        Careful with the regex method - it is able to match Wouter von Brno, but also Friendly Cast and Crew

        Args:
            email (Str): Email to be cleaned up
            methods (list of Str, optional): At least one of ['regex', 'email_adresses', 'databases' (not currently implemented)]. Can be either regex based (AGGRESIVE) or by finding names from email adresses included in the email text. TODO use McK database of names?. Defaults to ['email_adresses'].
            replacement (Str): What to replace the cleaned up name with
        Returns:
            email (Str): Cleaned up email
        """
        # TODO THINK ABOUT COMPANIES - maybe dont delete one letter names?
        allowed_methods = ['regex', 'email_adresses', 'database']
        for method in methods:
            assert method in allowed_methods, "Disallowed method! Must pass in list containing at least one of ['regex', 'email_adresses', 'database' (TODO)]"
        if 'regex' in methods:
            email = re.sub(r'[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z]([a-z]+|\.)',
                           replacement, email)
        if 'email_adresses' in methods:
            # needs to be called before removing names based on email adresses, otherwise this wont find anything!
            users = EmailCleaning.find_names_from_email_adresses(email)
            for user in users:
                # somes names get parsed out badly (or they are not in the email anymore) and it would error.. so we just skip those
                try:
                    desc = re.sub(r"\b%s\b" % user, replacement, desc)
                except:
                    print("Error when anonymizing:", user)
        # TODO MAYBE USE MCK DATABASE OF NAMES?
        if 'database' in methods:
            pass
        return email

    @staticmethod
    def remove_repeated_replies(email):
        """Useful when each of your emails containts the whole email thread.
        DANGEROUS: It matches on metadata and deletes everything onwards - it will fail if your email starts with metadata in the first place. But if its only after the real email, then it's good

        Args:
            email (Str): Email to be cleaned up

        Returns:
            email (Str): Cleaned up email
        """
        # this function finds some anchor used in signatures/replies and deletes everything from there onwards
        regex = []
        regex.append([r'On.*?wrote:[\s\S]*', ''])
        regex.append([r'am.*?schrieb:[\s\S]*', ''])
        regex.append([r'Le.*?écrit:[\s\S]*', ''])
        regex.append([r'Data:[\s\S]*', ''])
        regex.append([r'Datum:[\s\S]*', ''])
        regex.append([r'Sent:[\s\S]*', ''])
        regex.append([r'From:[\s\S]*', ''])
        regex.append([r'De:[\s\S]*', ''])
        regex.append([r'Sent from (my\s)?((blackberry)|(iphone)|(samsung)|(macbook)|(apple)|(pc))[\s\S]*', ''])
        regex.append([r'Von meinem ((blackberry)|(iphone)|(samsung)|(macbook)|(apple)|(pc))[\s\S]*', ''])
        regex.append([r'Subject:[\s\S]*', ''])
        # 08/20/2019 08:50 -> ''
        regex.append([r'\d{2}/\d{2}/\d{4}\s\d{2}[\s\S]*', ''])
        # the bars like | are only used in signatures
        regex.append([r'\│.*', ''])
        regex.append([
                         r'(McKinsey & Company|McKinsey & Co.|Cheers|Sincerely|Many thanks|Best Regards|kind regards|Regards|Warm regards|Assistant|Assistant:|Executive Assistant:)[\s\S]*',
                         ' '])
        for rg in regex:
            email = re.sub(rg[0], rg[1], email, flags=re.IGNORECASE | re.MULTILINE)
        return email

    @staticmethod
    def anonymize_urls(email, replacement='ANONYMIZED_URL', option='complex', disambiguate=[]):
        """Replaces URLs with placeholder. The 'complex' option is somewhat more reliable although extremely hard to edit
            I made the 'simple' option not require 'https' to make it match stuff like 'www.google.com'; no idea what else broke
        Args:
            email (Str): Email to be cleaned up
            replacement (str, optional): Defaults to 'ANONYMIZED_URL'.
            option (str, optional): One of ['simple', 'complex']. Defaults to 'simple'.
            disambiguate (list, optional): If not equal to [], disambiguates internal and external URLs. Defaults to [].

        Returns:
            email (Str): Cleaned up email
        """
        # The complex alternative is from https://gist.github.com/gruber/8891611 with Python-specific modifications. I also added the lookbehind in the beginning becaues it was matching mails, not sure if I broke it by accident while porting to Python
        # The simple alternative is https://stackoverflow.com/questions/6038061/regular-expression-to-find-urls-within-a-string
        alternative_regex = r'(?i)(?<!@)\b((?:https?:(?:\/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\\)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b\?(?!@)))'
        simple_alternative_regex = r'(http|ftp|https)?(://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'

        if option == 'simple':
            chosen_regex = simple_alternative_regex
        elif option == 'complex':
            chosen_regex = alternative_regex

        if disambiguate != []:
            matches = re.findall(chosen_regex, email)
            for match in matches:
                print(match)
                match = ''.join(match)
                print(match)
                if re.search('mckinsey.com', match) and 'internal' in disambiguate:
                    email = re.sub(match, replacement + 'INTERNAL', email)
                if not re.search('mckinsey.com', match) and 'external' in disambiguate:
                    email = re.sub(match, replacement + 'EXTERNAL', email)
        else:
            email = re.sub(chosen_regex, replacement, email)
        return email

    @staticmethod
    def anonymize_files(email, replacement='ANONYMIZED_FILE'):
        """ Replaces file names (identified by commmon file extensions) and
        power_of_love.pptx -> ANONYMIZED FILE
        Args:
            email (Str): Email to be cleaned up
            replacement (str, optional): Placeholder to replace the filename with. Defaults to 'ANONYMIZED_FILE'.

        Returns:
            email (Str): Cleaned up email
        """
        email = re.sub(
            r"([\w\d\-.]+\.pdf|[\w\d\-.]+\.docx|[\w\d\-.]+\.doc|[\w\d\-.]+\.pptx|[\w\d\-.]+\.ppt|[\w\d\-.]+\.txt|[\w\d\-.]+\.zip|[\w\d\-.]+\.xlsx|[\w\d\-.]+\.xls)",
            replacement, email)
        return email

    @staticmethod
    def anonymize_email_adresses(email, replacement='ANONYMIZED_EMAIL'):
        """Replaces email adresses with placeholder.

        Args:
            email (Str): Email to be cleaned up
            replacement (str, optional): Placeholder to replace the email with. Defaults to 'ANONYMIZED_EMAIL'.

        Returns:
            email (Str): Cleaned up email
        """
        # TODO allow to disambiguate between internal/external?
        regex = []
        # remove lotus notes email address
        regex.append([r'([a-z\-]+\s?\n?){1,3}(\/[a-z0-9\-?]+){2,5}(\@|\/)mckinsey(\-external|@M[a-z]*)?', replacement])
        # remove email address, john.snow@yahoo.com -> ANONYMIZED EMAIL
        regex.append([r'[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})', replacement])
        for rg in regex:
            email = re.sub(rg[0], rg[1], email, flags=re.IGNORECASE | re.MULTILINE)
        return email

    @staticmethod
    def find_names_from_email_adresses(email):
        """Extracts names of people who are sending messages within the email from stuff like From:first.last@yahoo.com
        Not sure how this works at this point to be honest
        Meant to be used with remove_names

        Args:
            email (Str): Email to be cleaned up

        Returns:
            (?): Not sure
        """
        senders = re.findall(r'\nFrom:([A-Za-z-\s]*)\n', email)
        from_names = list(map(lambda x: '_'.join(x.lower().split()), senders))

        email = email.lower()
        email = unidecode.unidecode(email)
        emails = re.findall(r"[a-z0-9\.\-+_,&:]+@[a-z0-9\.\-+_]+\.[a-z]+", email)

        users = list(np.unique(from_names + list(
            map(lambda x: x.replace('@external.mckinsey.com', '').replace('@mckinsey.com', ''), emails))))

        lower_case = users + [u.replace('_', ' ') for u in users] + [u.replace('_', '-') for u in users] + [
            u.split('_')[0] for u in users] + [u.split('_')[-1] for u in users] + [
                         u.split('_')[0] + ' ' + u.split('_')[-1] for u in users]
        lower_case = [l.strip().replace("mailto:", '') for l in lower_case if len(l.strip().replace("mailto:", '')) > 2]
        capitalized = list(np.unique(list(map(lambda x: x.title(), lower_case))))
        return lower_case + capitalized

    @staticmethod
    def insert_breaks(split_string,
                      delimiter_form='\n\n==============+++EMAIL_BREAK[PLACEHOLDER]+++==============\n\n'):
        """Interleaves split_string with delimiter_form (meant to then be concatenated into one thread string)

        Args:
            split_string (list of Str): List of individual emails
            delimiter_form (str, optional): Defaults to '\n\n==============+++EMAIL_BREAK[PLACEHOLDER]+++==============\n\n'.

        Returns:
            list of Str: Interleaved split_string
        """
        # the delimiter_form must have [PLACEHOLDER] if you want to have the delimiters intact after runing this on an email thread! this whole stuff is done so that np.unique in remove_repeating_parags does not remove the delimiters. putting {} into the default argument will not work
        new = []
        # delimiter='\n\n==============+++EMAIL_BREAK[PLACEHOLDER]+++==============\n\n'
        for i in range(len(split_string)):
            new.append(split_string[i])
            new.append(delimiter_form.replace('[PLACEHOLDER]', str(i)))
        return new[:-1]

    @staticmethod
    def remove_repeating_parags(thread, email_break='\n\n==============+++EMAIL_BREAK+++==============\n\n'):
        """MUST be used on whole email thread (otherwise, it won't contain repeated emails to delete)
        This is like a heuristic version of deleting emails that are being replied to in email threads (otherwise they woudl be there many times)

        Args:
            thread (Str): Email thread. Should contain some delimiters to separate individual emails
            email_break (str, optional): What is the delimiter between individual emails in a thread. Defaults to '\n\n==============+++EMAIL_BREAK+++==============\n\n'.

        Returns:
            thread (Str): Reconstructed email thread with paragraphs that repeated at some point removed apart from the first occurence (not sure here)
        """

        emails = thread.split(email_break)  # EmailCleaning.insert_breaks(thread.split(email_break), email_break)
        paragraphs = [' '.join(p.split()) for e in emails for p in re.split("\n{2,10}", e)]
        unique_parags, parags_idx = np.unique(paragraphs, return_index=True)
        non_repeating = EmailCleaning.remove_dups(unique_parags[np.argsort(parags_idx)])
        return ('\n\n').join(non_repeating)

    @staticmethod
    def remove_dups(unique_parags, email_break='\n\n==============+++EMAIL_BREAK+++==============\n\n'):
        """Helper function for remove_repeating_parags.

        Args:
            unique_parags (?): Unique paragraphs in the email
            email_break (str, optional): The delimiter used to separate emails in a thread. Defaults to '\n\n==============+++EMAIL_BREAK+++==============\n\n'.

        Returns:
            list: List of unique paragraphs (ordered) .. or something?
        """
        # as is, this has quadratic time complexity in pargraphs, but those tend to be few. could be made faster with OrderedDicts probably
        for_removal = []
        for u in range(len(unique_parags)):
            for i in range(u + 1, len(unique_parags)):
                if unique_parags[i] != '\n\n==============+++EMAIL_BREAK+++==============\n\n' and unique_parags[
                    u] != '\n\n==============+++EMAIL_BREAK+++==============\n\n':
                    if (unique_parags[i] in unique_parags[u]) or len(unique_parags[
                                                                         i]) < 30:  # (unique_parags[u] in unique_parags[i] or unique_parags[i] in unique_parags[u]) and len(unique_parags[u])>=35 and len(unique_parags[i])>=35:
                        for_removal.append(i)
        return [up for up in unique_parags[np.setdiff1d(range(len(unique_parags)), np.unique(for_removal))]]
